from argparse import ArgumentParser
import json
import os
import torch
import pytorch_lightning as pl
from src.models.patch import monkeypatch as make_functional
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from src.data.mmkb_dataset import MultihopCaptionDataset
from src.models.one_shot_learner import OneShotLearner
from src.utils import multiclass_log_probs
from src.models.get_models import get_model
from src.models.multimodal_training_hparams import MultimodalTrainingHparams


class MLLM_KE(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dev_data_path",
            type=str,
            default='/root/autodl-tmp/entity_level_eval.json'
        )
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--hop", type=str, choices=['1', '2', '3', '4'])
        parser.add_argument("--model_name", type=str, choices=["blip2", "minigpt4", "llava", "qwen-vl", "owl-2"], default="blip2")
        parser.add_argument("--model_checkpoint", type=str, required=True)
        return parser 
    


    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # self.model, self.tokenizer = get_model(self.hparams.model_name)
        if self.hparams.model_name == 'blip2' or self.hparams.model_name == 'minigpt4' or self.hparams.model_name == 'llava' :
            self.model, self.tokenizer = get_model(self.hparams.model_name)

        if self.hparams.model_name == 'blip2':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                'hparams/TRAINING/KE/blip2.yaml'
            )
            vocab_dim = self.model.opt_model.model.decoder.embed_tokens.weight.shape[0]
            embedding_dim = self.model.opt_model.model.decoder.embed_tokens.weight.shape[1]
            embedding_init = self.model.opt_model.model.decoder.embed_tokens.weight.data

        elif self.hparams.model_name == 'minigpt4':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                'hparams/TRAINING/KE/minigpt4.yaml'
            )
            vocab_dim = self.model.llama_model.model.embed_tokens.weight.shape[0]
            embedding_dim = self.model.llama_model.model.embed_tokens.weight.shape[1]
            embedding_init = self.model.llama_model.model.embed_tokens.weight.data
            
        elif self.hparams.model_name == 'llava':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                'hparams/TRAINING/KE/llava.yaml'
            )
            vocab_dim = self.model.model.embed_tokens.weight.shape[0]
            embedding_dim = self.model.model.embed_tokens.weight.shape[1]
            embedding_init = self.model.model.embed_tokens.weight.data

        elif self.hparams.model_name == 'qwen-vl':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                'hparams/TRAINING/KE/qwenvl.yaml'
            )
            self.model, self.tokenizer = get_model(self.model_hparams.name)
            vocab_dim = self.model.transformer.wte.weight.data.shape[0]
            embedding_dim = self.model.transformer.wte.weight.data.shape[1]
            embedding_init = self.model.transformer.wte.weight.data

        elif self.hparams.model_name == 'owl-2':
            self.model_hparams = MultimodalTrainingHparams.from_hparams(
                'hparams/TRAINING/KE/owl2.yaml'
            )
            self.model, self.tokenizer = get_model(self.model_hparams.name)
            vocab_dim = self.model.base_model.embed_tokens.weight.data.shape[0]
            embedding_dim = self.model.base_model.embed_tokens.weight.data.shape[1]
            embedding_init = self.model.base_model.embed_tokens.weight.data

        else:
            raise ValueError(f"Model {self.hparams.model_name} not supported")
        
        
        

        self.include_params_set={
            n
            for n, _ in self.model.named_parameters()
            if any(
                e in n.lower()
                for e in self.model_hparams.inner_params
            )
        }
        print(f"include_set: {self.include_params_set}")
        
        for n, p in self.model.named_parameters():
            if n in self.include_params_set:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.learner = OneShotLearner(
            self.model,
            vocab_dim=vocab_dim,
            embedding_dim=embedding_dim,
            hidden_dim=128,
            condition_dim=1024,
            include_set=self.include_params_set,
            max_scale=self.hparams.max_scale,
            embedding_init=embedding_init,
        ).to(torch.float32)

        self.valid_acc =    pl.metrics.Accuracy()
        self.valid_t_gen =  pl.metrics.Accuracy()
        self.valid_m_gen =  pl.metrics.Accuracy()
        self.valid_t_loc =  pl.metrics.Accuracy()
        self.valid_m_loc =  pl.metrics.Accuracy()
        self.valid_port =  pl.metrics.Accuracy()
         #新增
        self.valid_rel_1 =  pl.metrics.Accuracy()
        self.valid_rel_2 =  pl.metrics.Accuracy()
        self.valid_m_rel_1 =  pl.metrics.Accuracy()
        self.valid_m_rel_2 =  pl.metrics.Accuracy()
        self.valid_g_m_rel_1 =  pl.metrics.Accuracy()
        self.valid_g_m_rel_2 =  pl.metrics.Accuracy()
        #新增
        self.valid_rel =  pl.metrics.Accuracy()
        self.valid_m_rel =  pl.metrics.Accuracy()
        self.valid_g_m_rel =  pl.metrics.Accuracy()
        
        
        

        self.fmodel = None

        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.save_txt =  f'results/{cur_time}_{self.hparams.model_name}_port_hop{self.hparams.hop}.txt'
        self.save_json = f'results/{cur_time}_{self.hparams.model_name}_port_hop{self.hparams.hop}.json'
        self.port_result = []

    def val_dataloader(self, shuffle=False):
        if not hasattr(self, "val_dataset"):
            self.val_dataset = MultihopCaptionDataset(
                data_dir=self.hparams.dev_data_path, 
                config=self.model_hparams,
                hop=self.hparams.hop)
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def get_logits_orig_params_dict(self, batch):
        with torch.enable_grad():
            if self.hparams.model_name == "owl-2":
                input_ids, image = batch["edit_inner"]['input_ids'], batch["edit_inner"]['image']
                logit_for_grad= self.model.eval()(input_ids, 
                                            images=image.to(dtype=torch.float16)).logits
            else:
                logit_for_grad = self.model.eval()(
                    batch['edit_inner']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['edit_inner'],
                ).logits

            grads = torch.autograd.grad(
                multiclass_log_probs(
                    logit_for_grad,
                    batch['edit_inner']['labels']
                )["nll"],
                [p for n, p in self.model.named_parameters() if n in self.include_params_set],
            )
            grad_dict = {}
            for n, grad in zip([n for n, p in self.model.named_parameters() if n in self.include_params_set], grads):
                grad_dict[n] = grad

        params_dict = self.learner(
            batch['cond']["input_ids"],
            batch['cond']["attention_mask"],
            grads=grad_dict,
        )   

        return params_dict
    
    
    # def process_predict(self, logits, labels, tok):
        
    #     if logits.dim() == 3:
    #         # 移除最后一维并确保与labels的长度匹配
    #         logits = logits[:, :-1]
    #         logits = logits[:, -labels.shape[1]:]
        
    #     # 创建掩码并处理标签
    #     mask = labels != -100
    #     labels[~mask] = 0
    #     # 获取预测ID并根据掩码填充
    #     pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        
    #     predict = tok.decode(pred_ids.tolist()[0], skip_special_tokens=True)
    #     return predict
    
    def process_predict(self, logits, labels, tok):
        # 检查 logits 的维度
        if logits.dim() == 3:
            # 调整 logits 的最后一维，并确保与 labels 的长度一致
            logits = logits[:, :labels.shape[1]]  # 直接裁剪到和 labels 一致的长度

        # 创建掩码，排除 -100 的标签
        mask = labels != -100
        
        # 获取预测的类别 ID
        pred_ids = logits.argmax(-1)
        
        # 确保 pred_ids 和 mask 的维度一致
        if pred_ids.shape != mask.shape:
            raise ValueError(f"预测ID的形状 {pred_ids.shape} 和掩码的形状 {mask.shape} 不一致")

        # 根据掩码进行填充
        pred_ids = pred_ids.masked_fill(~mask, 0).detach().cpu()

        # 解码预测ID
        predict = tok.decode(pred_ids.tolist()[0], skip_special_tokens=True)
        
        return predict

    


    def validation_step(self, batch, batch_idx=None):
        assert len(batch['port']) == 1, "batch['port'] should have only one element"

        params_dict = self.get_logits_orig_params_dict(batch)
        self.fmodel = make_functional(self.model).eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                params=[params_dict.get(n, 0) + p for n, p in self.model.named_parameters()]
                
                
                
                def tensor_to_list(tensor):
                    return tensor.detach().cpu().numpy().tolist()  # 将Tensor转换为列表
                
                decode_output = {}
                
                logits = self.fmodel(batch['edit_inner'], params=params).logits
                results = multiclass_log_probs(logits, batch['edit_inner']['labels'])
                self.valid_acc(results["pred_ids"], results["targ_ids"])
                self.log("acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
                
                edit_inner_prompt = batch['edit_inner']['prompt']
                edit_inner_ground_truth = batch['edit_inner']['ground_truth']
                edit_inner_predict = self.process_predict(logits,batch['edit_inner']['labels'],self.tokenizer)
                decode_output['edit_inner_acc'] = tensor_to_list(results["acc"]) 
                decode_output['edit_inner_prompt'] = edit_inner_prompt
                decode_output['edit_inner_ground_truth'] = edit_inner_ground_truth
                decode_output['edit_inner_predict'] = edit_inner_predict


                logits = self.fmodel(batch['edit_outer'], params=params).logits
                results = multiclass_log_probs(logits, batch['edit_outer']['labels'])
                self.valid_acc(results["pred_ids"], results["targ_ids"])
                self.log("acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
                
                edit_outer_prompt = batch['edit_outer']['prompt']
                edit_outer_ground_truth = batch['edit_outer']['ground_truth']
                edit_outer_predict = self.process_predict(logits,batch['edit_outer']['labels'],self.tokenizer)
                decode_output['edit_outer_acc'] = tensor_to_list(results["acc"]) 
                decode_output['edit_outer_prompt'] = edit_outer_prompt
                decode_output['edit_outer_ground_truth'] = edit_outer_ground_truth
                decode_output['edit_outer_predict'] = edit_outer_predict



                logits = self.fmodel(batch['edit_outer_image'], params=params).logits
                results = multiclass_log_probs(logits, batch['edit_outer_image']['labels'])
                self.valid_m_gen(results["pred_ids"], results["targ_ids"])
                self.log("m_gen", self.valid_m_gen, on_step=False, on_epoch=True, prog_bar=True)
                
                edit_outer_image_prompt = batch['edit_outer_image']['prompt']
                edit_outer_image_ground_truth = batch['edit_outer_image']['ground_truth']
                edit_outer_image_predict = self.process_predict(logits,batch['edit_outer_image']['labels'],self.tokenizer)
                decode_output['edit_outer_image_acc'] = tensor_to_list(results["acc"]) 
                decode_output['edit_outer_image_prompt'] = edit_outer_image_prompt
                decode_output['edit_outer_image_ground_truth'] = edit_outer_image_ground_truth
                decode_output['edit_outer_image_predict'] = edit_outer_image_predict



                base_logits = self.model.eval()(batch["loc"]).logits
                base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices
                post_base_logits = self.fmodel(batch['loc'], params=params).logits
                post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                self.valid_t_loc(base_logits_softmax_top_k.view(-1), post_base_logits_softmax_top_k.view(-1))
                self.log("t_loc", self.valid_t_loc, on_step=False, on_epoch=True, prog_bar=True)
                
                loc_prompt = batch['loc']['prompt']
                loc_ground_truth = batch['loc']['ground_truth']
                loc_predict = self.process_predict(post_base_logits,batch['loc']['labels'],self.tokenizer)
                decode_output['loc_acc'] = tensor_to_list(results["acc"]) 
                decode_output['loc_prompt'] = loc_prompt
                decode_output['loc_ground_truth'] = loc_ground_truth
                decode_output['loc_predict'] = loc_predict
                
                base_image_logits = self.model.eval()(batch["loc_image"]).logits
                base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
                post_image_base_logits = self.fmodel(batch['loc_image'], params=params).logits
                post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
                self.valid_m_loc(base_image_logits_softmax_top_k.view(-1), post_image_base_logits_softmax_top_k.view(-1))
                self.log("m_loc", self.valid_m_loc, on_step=False, on_epoch=True, prog_bar=True)
                
                loc_image_prompt = batch['loc_image']['prompt']
                loc_image_ground_truth = batch['loc_image']['ground_truth']
                loc_image_predict = self.process_predict(post_base_logits,batch['loc_image']['labels'],self.tokenizer)
                decode_output['loc_image_acc'] = tensor_to_list(results["acc"]) 
                decode_output['loc_image_prompt'] = loc_image_prompt
                decode_output['loc_image_ground_truth'] = loc_image_ground_truth
                decode_output['loc_image_predict'] = loc_image_predict
                
                port = batch['port'][0]
                logits = self.fmodel(port, params=params).logits
                results = multiclass_log_probs(logits, port['labels'])
                self.valid_port(results["pred_ids"], results["targ_ids"])
                self.log("port", self.valid_port, on_step=False, on_epoch=True, prog_bar=True)
                
                port_prompt = port['prompt']
                port_ground_truth = port['ground_truth']
                port_predict = self.process_predict(logits,port['labels'],self.tokenizer)
                decode_output['port_acc'] = tensor_to_list(results["acc"]) 
                decode_output['port_prompt'] = port_prompt
                decode_output['port_ground_truth'] = port_ground_truth
                decode_output['port_predict'] = port_predict
                
                 ################################################
                if batch['knowledge_type']['knowledge_type']==0 or batch['knowledge_type']['knowledge_type']==1:
                    #文本问题
                    logits = self.fmodel(batch['T_Rel_1'], params=params).logits
                    results = multiclass_log_probs(logits, batch['T_Rel_1']['labels'])
                    self.valid_rel_1(results["pred_ids"], results["targ_ids"])
                    self.log("rel_1", self.valid_rel_1, on_step=False, on_epoch=True, prog_bar=True)
                    rel_1_prompt = batch['T_Rel_1']['prompt']
                    rel_1_ground_truth = batch['T_Rel_1']['ground_truth']
                    rel_1_predict = self.process_predict(logits,batch['T_Rel_1']['labels'],self.tokenizer)
                    decode_output['rel_1_acc'] = tensor_to_list(results["acc"]) 
                    decode_output['rel_1_prompt'] = rel_1_prompt
                    decode_output['rel_1_ground_truth'] = rel_1_ground_truth
                    decode_output['rel_1_predict'] = rel_1_predict

                    logits = self.fmodel(
                    batch['T_Rel_2']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['T_Rel_2'],
                    params=params,
                        ).logits
                    results = multiclass_log_probs(logits, batch['T_Rel_2']['labels'])
                    self.valid_rel_2(results["pred_ids"], results["targ_ids"])
                    self.log("rel_2", self.valid_rel_2, on_step=False, on_epoch=True, prog_bar=True)
                    rel_2_prompt = batch['T_Rel_2']['prompt']
                    rel_2_ground_truth = batch['T_Rel_2']['ground_truth']
                    rel_2_predict = self.process_predict(logits,batch['T_Rel_2']['labels'],self.tokenizer)
                    decode_output['rel_2_acc'] = tensor_to_list(results["acc"]) 
                    decode_output['rel_2_prompt'] = rel_2_prompt
                    decode_output['rel_2_ground_truth'] = rel_2_ground_truth
                    decode_output['rel_2_predict'] = rel_2_predict

                    
                    #多模态问题
                    logits = self.fmodel(
                    batch['M_Rel_1']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['M_Rel_1'],
                    params=params,
                        ).logits
                    results = multiclass_log_probs(logits, batch['M_Rel_1']['labels'])
                    self.valid_m_rel_1(results["pred_ids"], results["targ_ids"])
                    self.log("m_rel_1", self.valid_m_rel_1, on_step=False, on_epoch=True, prog_bar=True)
                    m_rel_1_prompt = batch['M_Rel_1']['prompt']
                    m_rel_1_ground_truth = batch['M_Rel_1']['ground_truth']
                    m_rel_1_predict = self.process_predict(logits,batch['M_Rel_1']['labels'],self.tokenizer)
                    decode_output['m_rel_1_acc'] = tensor_to_list(results["acc"]) 
                    decode_output['m_rel_1_prompt'] = m_rel_1_prompt
                    decode_output['m_rel_1_ground_truth'] = m_rel_1_ground_truth
                    decode_output['m_rel_1_predict'] = m_rel_1_predict
                    
            
                    
                    logits = self.fmodel(
                    batch['M_Rel_2']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['M_Rel_2'],
                    params=params,
                        ).logits
                    results = multiclass_log_probs(logits, batch['M_Rel_2']['labels'])
                    self.valid_m_rel_2(results["pred_ids"], results["targ_ids"])
                    self.log("m_rel_2", self.valid_m_rel_2, on_step=False, on_epoch=True, prog_bar=True)
                    m_rel_2_prompt = batch['M_Rel_2']['prompt']
                    m_rel_2_ground_truth = batch['M_Rel_2']['ground_truth']
                    m_rel_2_predict = self.process_predict(logits,batch['M_Rel_2']['labels'],self.tokenizer)
                    decode_output['m_rel_2_acc'] = tensor_to_list(results["acc"]) 
                    decode_output['m_rel_2_prompt'] = m_rel_2_prompt
                    decode_output['m_rel_2_ground_truth'] = m_rel_2_ground_truth
                    decode_output['m_rel_2_predict'] = m_rel_2_predict
                    
                
                    
                    #多模态问题泛化
                    logits = self.fmodel(
                    batch['Gen_M_Rel_1']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['Gen_M_Rel_1'],
                    params=params,
                        ).logits
                    results = multiclass_log_probs(logits, batch['Gen_M_Rel_1']['labels'])
                    self.valid_g_m_rel_1(results["pred_ids"], results["targ_ids"])
                    self.log("g_m_rel_1", self.valid_g_m_rel_1, on_step=False, on_epoch=True, prog_bar=True)
                    g_m_rel_1_prompt = batch['Gen_M_Rel_1']['prompt']
                    g_m_rel_1_ground_truth = batch['Gen_M_Rel_1']['ground_truth']
                    g_m_rel_1_predict = self.process_predict(logits,batch['Gen_M_Rel_1']['labels'],self.tokenizer)
                    decode_output['g_m_rel_1_acc'] = tensor_to_list(results["acc"]) 
                    decode_output['g_m_rel_1_prompt'] = g_m_rel_1_prompt
                    decode_output['g_m_rel_1_ground_truth'] = g_m_rel_1_ground_truth
                    decode_output['g_m_rel_1_predict'] = g_m_rel_1_predict

                    logits = self.fmodel(
                    batch['Gen_M_Rel_2']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['Gen_M_Rel_2'],
                    params=params,
                        ).logits
                    results = multiclass_log_probs(logits, batch['Gen_M_Rel_2']['labels'])
                    self.valid_g_m_rel_2(results["pred_ids"], results["targ_ids"])
                    self.log("g_m_rel_2", self.valid_g_m_rel_2, on_step=False, on_epoch=True, prog_bar=True)
                    g_m_rel_2_prompt = batch['Gen_M_Rel_2']['prompt']
                    g_m_rel_2_ground_truth = batch['Gen_M_Rel_2']['ground_truth']
                    g_m_rel_2_predict = self.process_predict(logits,batch['Gen_M_Rel_2']['labels'],self.tokenizer)
                    decode_output['g_m_rel_2_acc'] = tensor_to_list(results["acc"]) 
                    decode_output['g_m_rel_2_prompt'] = g_m_rel_2_prompt
                    decode_output['g_m_rel_2_ground_truth'] = g_m_rel_2_ground_truth
                    decode_output['g_m_rel_2_predict'] = g_m_rel_2_predict
                    
                    decode_output['knowledge_type'] =batch['knowledge_type']['knowledge_type']
                  
                    
                elif batch['knowledge_type']['knowledge_type']== 2:
                    #文本问题
                    logits = self.fmodel(
                    batch['T_Rel']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['T_Rel'],
                    params=params,
                        ).logits
                    results = multiclass_log_probs(logits, batch['T_Rel']['labels'])
                    self.valid_rel(results["pred_ids"], results["targ_ids"])
                    self.log("rel", self.valid_rel, on_step=False, on_epoch=True, prog_bar=True)
                    rel_prompt = batch['T_Rel']['prompt']
                    rel_ground_truth = batch['T_Rel']['ground_truth']
                    rel_predict = self.process_predict(logits,batch['T_Rel']['labels'],self.tokenizer)
                    decode_output['rel_acc'] = tensor_to_list(results["acc"]) 
                    decode_output['rel_prompt'] = rel_prompt
                    decode_output['rel_ground_truth'] = rel_ground_truth
                    decode_output['rel_predict'] = rel_predict
         
                    
                    #多模态问题
                    logits = self.fmodel(
                    batch['M_Rel']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['M_Rel'],
                    params=params,
                        ).logits
                    results = multiclass_log_probs(logits, batch['M_Rel']['labels'])
                    self.valid_m_rel(results["pred_ids"], results["targ_ids"])
                    self.log("m_rel", self.valid_m_rel, on_step=False, on_epoch=True, prog_bar=True)
                    m_rel_prompt = batch['M_Rel']['prompt']
                    m_rel_ground_truth = batch['M_Rel']['ground_truth']
                    m_rel_predict = self.process_predict(logits,batch['M_Rel']['labels'],self.tokenizer)
                    decode_output['m_rel_acc'] = tensor_to_list(results["acc"]) 
                    decode_output['m_rel_prompt'] = m_rel_prompt
                    decode_output['m_rel_ground_truth'] = m_rel_ground_truth
                    decode_output['m_rel_predict'] = m_rel_predict
        
                    
                    #多模态问题泛化
                    logits = self.fmodel(
                    batch['Gen_M_Rel']['inputs'] if self.hparams.model_name == "qwen-vl" else batch['Gen_M_Rel'],
                    params=params,
                        ).logits
                    results = multiclass_log_probs(logits, batch['Gen_M_Rel']['labels'])
                    self.valid_g_m_rel(results["pred_ids"], results["targ_ids"])
                    self.log("g_m_rel", self.valid_g_m_rel, on_step=False, on_epoch=True, prog_bar=True)
                    g_m_rel_prompt = batch['Gen_M_Rel']['prompt']
                    g_m_rel_ground_truth = batch['Gen_M_Rel']['ground_truth']
                    g_m_rel_predict = self.process_predict(logits,batch['Gen_M_Rel']['labels'],self.tokenizer)
                    
                    decode_output['g_m_rel_acc'] = tensor_to_list(results["acc"]) 
                    decode_output['g_m_rel_prompt'] = g_m_rel_prompt
                    decode_output['g_m_rel_ground_truth'] = g_m_rel_ground_truth
                    decode_output['g_m_rel_predict'] = g_m_rel_predict
                    
                    decode_output['knowledge_type'] =batch['knowledge_type']['knowledge_type']
      
                    ###############################################
                from datetime import datetime
                cur_time = datetime.now().strftime("%y%m%d_%H%M%S")          
                knowledge_type = decode_output['knowledge_type']
                if decode_output['knowledge_type']==0:
                    knowledge_type = 'entity-level'
                elif decode_output['knowledge_type']==1:
                    knowledge_type = 'user-specific'
                elif decode_output['knowledge_type']==2:
                    knowledge_type = 'visual-knowledge'
                    
                # file_output = f'{cur_time}_{self.hparams.model_name}_{knowledge_type}_KE_result.json'
                file_output = f'{self.hparams.model_name}_{knowledge_type}_KE_result.json'
                if os.path.exists(file_output):
                    # 如果文件存在，读取现有数据
                    with open(file_output, 'r') as file:
                        try:
                            data = json.load(file)  # 读取 JSON 文件中的列表
                            if not isinstance(data, list):
                                data = []  # 如果文件中不是列表，初始化为空列表
                        except json.JSONDecodeError:
                            data = []  # 如果文件为空或解析错误，初始化为空列表
                else:
                    data = []  # 如果文件不存在，初始化为空列表

                # 将新的 decode_output 追加到列表中
                data.append(decode_output)

                # 将整个列表重新写回文件
                with open(file_output, 'w') as file:
                    json.dump(data, file, indent=4)








                # edit_inputs = batch['edit_inner']['text_input']
                # port_inputs = batch['port'][0]['text_input']
                # port_acc = results['acc'].item()
                # port_pred_ids = results['pred_ids'].cpu().numpy().tolist()
                # port_targ_ids = results['targ_ids'].cpu().numpy().tolist()

                # with open(f'{self.save_txt}', 'a') as f:
                #     f.write(f'{edit_inputs}\n{port_inputs}\n{port_acc}\npred: {port_pred_ids}\ntarget: {port_targ_ids}\n\n')
                # self.port_result.append({
                #     'edit_input': edit_inputs,
                #     'port_input': port_inputs,
                #     'port_acc': port_acc,
                #     'port_pred_ids': port_pred_ids,
                #     'port_targ_ids': port_targ_ids
                # })

    def on_validation_epoch_end(self) -> None:
        self.fmodel = None
        # with open(f'{self.save_json}', 'w') as f:
        #     json.dump(self.port_result, f, indent=2)
        return super().on_validation_epoch_end()
