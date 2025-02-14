from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)
from tqdm import tqdm
from transformers import AutoTokenizer

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]


        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            if config.tokenizer_class == "QWenTokenizer":
                tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True, pad_token='<|endoftext|>')
            elif config.model_name == "owl-2":
                tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=False, trust_remote_code=True)
            else:
                tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                    tok_name, trust_remote_code=True
                )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token 
        
        self.tok = tokenizer 

    def process_predict(self, logits, labels, tok):
        
        if logits.dim() == 3:
            logits = logits[:, :-1]
            logits = logits[:, -labels.shape[1]:]
        
        
        mask = labels != -100
        labels[~mask] = 0
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        predict = tok.decode(pred_ids.tolist()[0], skip_special_tokens=True)
        return predict



    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)
        tok = self.tok
        ####################################################################################################
        with torch.no_grad():
            base_outputs = self.model(batch["loc"])
            if not isinstance(base_outputs, torch.Tensor):
                base_logits = base_outputs.logits
            else:  
                base_logits = base_outputs
            del base_outputs
            torch.cuda.empty_cache()
                
            base_image_outputs = self.model(batch["loc_image"])
            if not isinstance(base_image_outputs, torch.Tensor):
                base_image_logits = base_image_outputs.logits
            else:
                base_image_logits = base_image_outputs
            del base_image_outputs
            torch.cuda.empty_cache()
        ####################################################################################################


        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0 
        info_dict = {}


        ####################################################################################################
        with torch.set_grad_enabled(training):
            # rephrase prompt
            post_edit_outputs = edited_model(batch["edit_outer"])
            post_batch_labels = batch["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs
            
            edit_outer_prompt = batch["edit_outer"]['prompt']
            edit_outer_ground_truth = batch["edit_outer"]['ground_truth']
            edit_outer_predict = self.process_predict(post_edit_logits,post_batch_labels,tok)
            
            del post_edit_outputs
            torch.cuda.empty_cache()


            # rephrase image
            post_image_edit_outputs = edited_model(batch["edit_outer_image"])
            post_image_batch_labels = batch["edit_outer_image"]["labels"]

            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs
       
            edit_outer_image_prompt = batch["edit_outer_image"]['prompt']
            edit_outer_image_ground_truth = batch["edit_outer_image"]['ground_truth']
            edit_outer_image_predict = self.process_predict(post_image_edit_logits,post_image_batch_labels,tok)
            
            del post_image_edit_outputs
            torch.cuda.empty_cache()
            
            # prompt           
            inner_edit_outputs = edited_model(batch["edit_inner"])
            inner_batch_labels = batch["edit_inner"]["labels"]

            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs
                       
            edit_inner_prompt = batch["edit_inner"]['prompt']
            edit_inner_ground_truth = batch["edit_inner"]['ground_truth']
            edit_inner_predict = self.process_predict(inner_edit_logits,inner_batch_labels,tok)
            del inner_edit_outputs
            torch.cuda.empty_cache()

            # rephrase prompt
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)["nll"]
            else:
                l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])["nll"]

            with torch.no_grad():
                if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
                else:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])
                
                del post_edit_logits,post_batch_labels
                torch.cuda.empty_cache()

            # rephrase image
            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)["nll"]
            else:
                l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])["nll"]

            with torch.no_grad():
                if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
                    image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)
                else:
                    image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
                del post_image_edit_logits,post_image_batch_labels
                torch.cuda.empty_cache()
                
                # prompt  
                if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
                else:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])
                
                del inner_edit_logits,inner_batch_labels
                torch.cuda.empty_cache()
                

        
            # post loc 
            post_base_outputs = edited_model(batch["loc"])
            post_base_label = batch["loc"]["labels"]
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
                kl_mask = post_base_outputs.attention_mask
            else:
                post_base_logits = post_base_outputs
                kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

            loc_prompt = batch["loc"]['prompt']
            loc_ground_truth = batch["loc"]['ground_truth']
            loc_predict = self.process_predict(post_base_logits,post_base_label,tok)
            del post_base_label
            torch.cuda.empty_cache()

            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)

            # post image loc 
            post_image_base_outputs = edited_model(batch["loc_image"])
            post_image_base_label =  batch["loc_image"]["labels"]
            if not isinstance(post_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
                kl_image_mask = post_image_base_outputs.attention_mask
            else:
                post_image_base_logits = post_image_base_outputs
                kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(base_image_logits.device)

            del post_image_base_outputs,post_base_outputs
            torch.cuda.empty_cache()


            loc_image_prompt = batch["loc_image"]['prompt']
            loc_image_ground_truth = batch["loc_image"]['ground_truth']
            loc_image_predict = self.process_predict(post_image_base_logits,post_image_base_label,tok)

            del post_image_base_label
            torch.cuda.empty_cache()

            l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

        # Text locality
        post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
        base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

        del post_base_logits,base_logits
        torch.cuda.empty_cache()

        # Image locality
        post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
        base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices
        
        del post_image_base_logits,base_image_logits
        torch.cuda.empty_cache()

        with torch.set_grad_enabled(training):
            #################################################################
            if "T_Rel_1" in batch:
                #T-Rel_1
                T_Rel_1_edit_outputs = edited_model(batch["T_Rel_1"])
                T_Rel_1_batch_labels = batch["T_Rel_1"]["labels"]

                if not isinstance(T_Rel_1_edit_outputs, torch.Tensor):
                    T_Rel_1_edit_logits = T_Rel_1_edit_outputs.logits
                else:
                    T_Rel_1_edit_logits = T_Rel_1_edit_outputs

                del T_Rel_1_edit_outputs
                torch.cuda.empty_cache()

                T_Rel_1_prompt = batch["T_Rel_1"]['prompt']
                T_Rel_1_ground_truth = batch["T_Rel_1"]['ground_truth']
                T_Rel_1_predict = self.process_predict(T_Rel_1_edit_logits,T_Rel_1_batch_labels,tok)

                with torch.no_grad():
                    if T_Rel_1_edit_logits.shape[1] > T_Rel_1_batch_labels.shape[1]:
                        T_Rel_1_edit_dict = self.model.edit_loss_fn(self.config, T_Rel_1_edit_logits, T_Rel_1_batch_labels)
                    else:
                        T_Rel_1_edit_dict = self.model.edit_loss_fn(self.config, T_Rel_1_edit_logits, T_Rel_1_batch_labels[:, -T_Rel_1_edit_logits.shape[1]-1:])
                    del T_Rel_1_edit_logits
                    torch.cuda.empty_cache()


                # T_Rel_2
                T_Rel_2_edit_outputs = edited_model(batch["T_Rel_2"])
                T_Rel_2_batch_labels = batch["T_Rel_2"]["labels"]
                if not isinstance(T_Rel_2_edit_outputs, torch.Tensor):
                    T_Rel_2_edit_logits = T_Rel_2_edit_outputs.logits
                else:
                    T_Rel_2_edit_logits = T_Rel_2_edit_outputs
 
                del T_Rel_2_edit_outputs
                torch.cuda.empty_cache()
                T_Rel_2_prompt = batch["T_Rel_2"]['prompt']
                T_Rel_2_ground_truth = batch["T_Rel_2"]['ground_truth']
                T_Rel_2_predict = self.process_predict(T_Rel_2_edit_logits,T_Rel_2_batch_labels,tok)

                with torch.no_grad():
                    if T_Rel_2_edit_logits.shape[1] > T_Rel_2_batch_labels.shape[1]:
                        T_Rel_2_edit_dict = self.model.edit_loss_fn(self.config, T_Rel_2_edit_logits, T_Rel_2_batch_labels)
                    else:
                        T_Rel_2_edit_dict = self.model.edit_loss_fn(self.config, T_Rel_2_edit_logits, T_Rel_2_batch_labels[:, -T_Rel_2_edit_logits.shape[1]-1:])
                    del T_Rel_2_edit_logits
                    torch.cuda.empty_cache()

                # M_Rel_1
                M_Rel_1_edit_outputs = edited_model(batch["M_Rel_1"])
                M_Rel_1_batch_labels = batch["M_Rel_1"]["labels"]

                if not isinstance(M_Rel_1_edit_outputs, torch.Tensor):
                    M_Rel_1_edit_logits = M_Rel_1_edit_outputs.logits
                else:
                    M_Rel_1_edit_logits = M_Rel_1_edit_outputs

                del M_Rel_1_edit_outputs
                torch.cuda.empty_cache()

                M_Rel_1_prompt = batch["M_Rel_1"]['prompt']
                M_Rel_1_ground_truth = batch["M_Rel_1"]['ground_truth']
                M_Rel_1_predict = self.process_predict(M_Rel_1_edit_logits,M_Rel_1_batch_labels,tok)
                
                with torch.no_grad():                        
                    if M_Rel_1_edit_logits.shape[1] > M_Rel_1_batch_labels.shape[1]:
                        M_Rel_1_edit_dict = self.model.edit_loss_fn(self.config, M_Rel_1_edit_logits, M_Rel_1_batch_labels)
                    else:
                        M_Rel_1_edit_dict = self.model.edit_loss_fn(self.config, M_Rel_1_edit_logits, M_Rel_1_batch_labels[:, -M_Rel_1_edit_logits.shape[1]-1:])
                    del M_Rel_1_edit_logits
                    torch.cuda.empty_cache()

                # M_Rel_2
                M_Rel_2_edit_outputs = edited_model(batch["M_Rel_2"])
                M_Rel_2_batch_labels = batch["M_Rel_2"]["labels"]

                if not isinstance(M_Rel_2_edit_outputs, torch.Tensor):
                    M_Rel_2_edit_logits = M_Rel_2_edit_outputs.logits
                else:
                    M_Rel_2_edit_logits = M_Rel_2_edit_outputs

                del M_Rel_2_edit_outputs
                torch.cuda.empty_cache()
                M_Rel_2_prompt = batch["M_Rel_2"]['prompt']
                M_Rel_2_ground_truth = batch["M_Rel_2"]['ground_truth']
                M_Rel_2_predict = self.process_predict(M_Rel_2_edit_logits,M_Rel_2_batch_labels,tok)

                with torch.no_grad():  
                    if M_Rel_2_edit_logits.shape[1] > M_Rel_2_batch_labels.shape[1]:
                        M_Rel_2_edit_dict = self.model.edit_loss_fn(self.config, M_Rel_2_edit_logits, M_Rel_2_batch_labels)
                    else:
                        M_Rel_2_edit_dict = self.model.edit_loss_fn(self.config, M_Rel_2_edit_logits, M_Rel_2_batch_labels[:, -M_Rel_2_edit_logits.shape[1]-1:])
                    del M_Rel_2_edit_logits
                    torch.cuda.empty_cache()


                # Gen_M_Rel_1
                Gen_M_Rel_1_edit_outputs = edited_model(batch["Gen_M_Rel_1"])
                Gen_M_Rel_1_batch_labels = batch["Gen_M_Rel_1"]["labels"]
                

                if not isinstance(Gen_M_Rel_1_edit_outputs, torch.Tensor):
                    Gen_M_Rel_1_edit_logits = Gen_M_Rel_1_edit_outputs.logits
                else:
                    Gen_M_Rel_1_edit_logits = Gen_M_Rel_1_edit_outputs
                
                del Gen_M_Rel_1_edit_outputs
                torch.cuda.empty_cache()
                Gen_M_Rel_1_prompt = batch["Gen_M_Rel_1"]['prompt']
                Gen_M_Rel_1_ground_truth = batch["Gen_M_Rel_1"]['ground_truth']
                Gen_M_Rel_1_predict = self.process_predict(Gen_M_Rel_1_edit_logits,Gen_M_Rel_1_batch_labels,tok)
                
                with torch.no_grad():  
                
                    if Gen_M_Rel_1_edit_logits.shape[1] > Gen_M_Rel_1_batch_labels.shape[1]:
                        Gen_M_Rel_1_edit_dict = self.model.edit_loss_fn(self.config, Gen_M_Rel_1_edit_logits, Gen_M_Rel_1_batch_labels)
                    else:
                        Gen_M_Rel_1_edit_dict = self.model.edit_loss_fn(self.config, Gen_M_Rel_1_edit_logits, Gen_M_Rel_1_batch_labels[:, -Gen_M_Rel_1_edit_logits.shape[1]-1:])
                    del Gen_M_Rel_1_edit_logits
                    torch.cuda.empty_cache()
                
                # Gen_M_Rel_2
                Gen_M_Rel_2_edit_outputs = edited_model(batch["Gen_M_Rel_2"])
                Gen_M_Rel_2_batch_labels = batch["Gen_M_Rel_2"]["labels"]
  
                if not isinstance(Gen_M_Rel_2_edit_outputs, torch.Tensor):
                    Gen_M_Rel_2_edit_logits = Gen_M_Rel_2_edit_outputs.logits
                else:
                    Gen_M_Rel_2_edit_logits = Gen_M_Rel_2_edit_outputs
                                
                del Gen_M_Rel_2_edit_outputs
                torch.cuda.empty_cache()
                Gen_M_Rel_2_prompt = batch["Gen_M_Rel_2"]['prompt']
                Gen_M_Rel_2_ground_truth = batch["Gen_M_Rel_2"]['ground_truth']
                Gen_M_Rel_2_predict = self.process_predict(Gen_M_Rel_2_edit_logits,Gen_M_Rel_2_batch_labels,tok)

                with torch.no_grad():  
                    if Gen_M_Rel_2_edit_logits.shape[1] > Gen_M_Rel_2_batch_labels.shape[1]:
                        Gen_M_Rel_2_edit_dict = self.model.edit_loss_fn(self.config, Gen_M_Rel_2_edit_logits, Gen_M_Rel_2_batch_labels)
                    else:
                        Gen_M_Rel_2_edit_dict = self.model.edit_loss_fn(self.config, Gen_M_Rel_2_edit_logits, Gen_M_Rel_2_batch_labels[:, -Gen_M_Rel_2_edit_logits.shape[1]-1:])
                    del Gen_M_Rel_2_edit_logits
                    torch.cuda.empty_cache()

            else:
                # T-Rel
                T_Rel_edit_outputs = edited_model(batch["T_Rel"])
                T_Rel_batch_labels = batch["T_Rel"]["labels"]

                if not isinstance(T_Rel_edit_outputs, torch.Tensor):
                    T_Rel_edit_logits = T_Rel_edit_outputs.logits
                else:
                    T_Rel_edit_logits = T_Rel_edit_outputs
                                                
                del T_Rel_edit_outputs
                torch.cuda.empty_cache()
                T_Rel_prompt = batch["T_Rel"]['prompt']
                T_Rel_ground_truth = batch["T_Rel"]['ground_truth']
                T_Rel_predict = self.process_predict(T_Rel_edit_logits,T_Rel_batch_labels,tok)
                
                with torch.no_grad():  
                    if T_Rel_edit_logits.shape[1] > T_Rel_batch_labels.shape[1]:
                        T_Rel_edit_dict = self.model.edit_loss_fn(self.config, T_Rel_edit_logits, T_Rel_batch_labels)
                    else:
                        T_Rel_edit_dict = self.model.edit_loss_fn(self.config, T_Rel_edit_logits, T_Rel_batch_labels[:, -T_Rel_edit_logits.shape[1]-1:])
                    del T_Rel_edit_logits
                    torch.cuda.empty_cache()

                # M_Rel
                M_Rel_edit_outputs = edited_model(batch["M_Rel"])
                M_Rel_batch_labels = batch["M_Rel"]["labels"]

                if not isinstance(M_Rel_edit_outputs, torch.Tensor):
                    M_Rel_edit_logits = M_Rel_edit_outputs.logits
                else:
                    M_Rel_edit_logits = M_Rel_edit_outputs
                del M_Rel_edit_outputs
                torch.cuda.empty_cache()
                M_Rel_prompt = batch["M_Rel"]['prompt']
                M_Rel_ground_truth = batch["M_Rel"]['ground_truth']
                M_Rel_predict = self.process_predict(M_Rel_edit_logits,M_Rel_batch_labels,tok)
                
                with torch.no_grad():  
                    if M_Rel_edit_logits.shape[1] > M_Rel_batch_labels.shape[1]:
                        M_Rel_edit_dict = self.model.edit_loss_fn(self.config, M_Rel_edit_logits, M_Rel_batch_labels)
                    else:
                        M_Rel_edit_dict = self.model.edit_loss_fn(self.config, M_Rel_edit_logits, M_Rel_batch_labels[:, -M_Rel_edit_logits.shape[1]-1:])
                    del M_Rel_edit_logits
                    torch.cuda.empty_cache()

                # Gen_M_Rel
                Gen_M_Rel_edit_outputs = edited_model(batch["Gen_M_Rel"])
                Gen_M_Rel_batch_labels = batch["Gen_M_Rel"]["labels"]

                if not isinstance(Gen_M_Rel_edit_outputs, torch.Tensor):
                    Gen_M_Rel_edit_logits = Gen_M_Rel_edit_outputs.logits
                else:
                    Gen_M_Rel_edit_logits = Gen_M_Rel_edit_outputs
                                              
                del Gen_M_Rel_edit_outputs
                torch.cuda.empty_cache()
                Gen_M_Rel_prompt = batch["Gen_M_Rel"]['prompt']
                Gen_M_Rel_ground_truth = batch["Gen_M_Rel"]['ground_truth']
                Gen_M_Rel_predict = self.process_predict(Gen_M_Rel_edit_logits,Gen_M_Rel_batch_labels,tok)

                with torch.no_grad():  
                    if Gen_M_Rel_edit_logits.shape[1] > Gen_M_Rel_batch_labels.shape[1]:
                        Gen_M_Rel_edit_dict = self.model.edit_loss_fn(self.config, Gen_M_Rel_edit_logits, Gen_M_Rel_batch_labels)
                    else:
                        Gen_M_Rel_edit_dict = self.model.edit_loss_fn(self.config, Gen_M_Rel_edit_logits, Gen_M_Rel_batch_labels[:, -Gen_M_Rel_edit_logits.shape[1]-1:])
                    del Gen_M_Rel_edit_logits
                    torch.cuda.empty_cache()
            #################################################################


        


        if l_edit.isnan():
            print("l_edit is nan")
            print("input: ", batch["edit_outer"]['text_input'])
        elif l_image_edit.isnan():
            print("l_image_edit is nan")
            print("input: ", batch["edit_outer_image"]['text_input'])
        elif l_loc.isnan():
            print("l_loc is nan")
            print("input: ", batch["loc"]['text_input'])
        elif l_image_loc.isnan():
            print("l_image_loc is nan")
            print("input: ", batch["loc_image"]['text_input'])


            
        if self.config.alg == "SERAC_MULTI":
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + self.config.iedit * l_image_edit
        else:
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc+l_image_loc) + self.config.iedit * l_image_edit
        

        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)



        info_dict['loss/edit'] = l_edit.item()
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()
        info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict["time/edit"] = edit_time
        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base
        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]

        #####################################
        # if batch['knowledge_type']['knowledge_type'] == 0 or batch['knowledge_type']['knowledge_type']==1:
        if "T_Rel_1" in batch:
           
            info_dict["T_Rel_1/acc"]= T_Rel_1_edit_dict["acc"].item()
            info_dict["T_Rel_2/acc"]= T_Rel_2_edit_dict["acc"].item()
            info_dict["T_Rel_average/acc"]= (info_dict["T_Rel_1/acc"] + info_dict["T_Rel_2/acc"]) / 2
            
            info_dict["M_Rel_1/acc"]= M_Rel_1_edit_dict["acc"].item()
            info_dict["M_Rel_2/acc"]= M_Rel_2_edit_dict["acc"].item()
            info_dict["M_Rel_average/acc"]= (info_dict["M_Rel_1/acc"] + info_dict["M_Rel_2/acc"]) / 2
           
            info_dict["Gen_M_Rel_1/acc"]= Gen_M_Rel_1_edit_dict["acc"].item()
            info_dict["Gen_M_Rel_2/acc"]= Gen_M_Rel_2_edit_dict["acc"].item()
            info_dict["Gen_M_Rel_average/acc"]= (info_dict["Gen_M_Rel_1/acc"] + info_dict["Gen_M_Rel_2/acc"]) / 2

            if batch['knowledge_type']['knowledge_type']== 0:
                info_dict['knowledge_type'] = 0
            elif batch['knowledge_type']['knowledge_type']== 1:
                info_dict['knowledge_type'] = 1

        else:
            
            info_dict["T_Rel/acc"]= T_Rel_edit_dict["acc"].item()
            info_dict["M_Rel/acc"]= M_Rel_edit_dict["acc"].item()
            info_dict["Gen_M_Rel/acc"]= Gen_M_Rel_edit_dict["acc"].item()

            info_dict['knowledge_type'] =2
        #####################################


        ####################################################################################################

        ################ portability #################
        if batch['port'] is not None:
            port_acc = 0
            for port in batch['port']:
                with torch.no_grad():
                    port_outputs = edited_model(port)
                    port_labels = port["labels"]
                    if not isinstance(port_outputs, torch.Tensor):
                        port_logits = port_outputs.logits
                    else:
                        port_logits = port_outputs
                    
                    del port_outputs
                    torch.cuda.empty_cache()
                    
                    Port_prompt = batch['port'][0]['prompt']
                    Port_ground_truth = batch['port'][0]['ground_truth']
                    Port_predict = self.process_predict(port_logits,port_labels,tok)

                    if port_logits.shape[1] > port_labels.shape[1]:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels)
                    else:
                        port_dict = self.model.edit_loss_fn(self.config, port_logits, port_labels[:, -port_logits.shape[1]-1:])
                    
                    del port_labels,port_logits
                    torch.cuda.empty_cache()
                    
                    port_acc += port_dict["acc"].item()
            port_acc /= len(batch['port'])
            info_dict['port/acc'] = port_acc
            del port_acc
            torch.cuda.empty_cache()
            info_dict['Port_prompt'] = Port_prompt
            del Port_prompt
            torch.cuda.empty_cache()
            info_dict['Port_ground_truth'] = Port_ground_truth
            del Port_ground_truth
            torch.cuda.empty_cache()
            info_dict['Port_predict'] = Port_predict
            del Port_predict
            torch.cuda.empty_cache()
        ################ portability #################
        
        info_dict['edit_inner_prompt'] = edit_inner_prompt
        info_dict['edit_inner_ground_truth'] = edit_inner_ground_truth
        info_dict['inner_predict'] = edit_inner_predict
        del edit_inner_prompt,edit_inner_ground_truth,edit_inner_predict
        torch.cuda.empty_cache()


        info_dict['edit_outer_prompt'] = edit_outer_prompt
        info_dict['edit_outer_ground_truth'] = edit_outer_ground_truth
        info_dict['edit_outer_predict'] = edit_outer_predict
        del edit_outer_prompt,edit_outer_ground_truth,edit_outer_predict
        torch.cuda.empty_cache()
        
        info_dict['edit_outer_image_prompt'] = edit_outer_image_prompt
        info_dict['edit_outer_image_ground_truth'] = edit_outer_image_ground_truth
        info_dict['edit_outer_image_predict'] = edit_outer_image_predict
        del edit_outer_image_prompt,edit_outer_image_ground_truth,edit_outer_image_predict
        torch.cuda.empty_cache()
        
        info_dict['loc_prompt'] = loc_prompt
        info_dict['loc_ground_truth'] = loc_ground_truth
        info_dict['loc_predict'] = loc_predict
        del loc_prompt,loc_ground_truth,loc_predict
        torch.cuda.empty_cache()
        
        info_dict['loc_image_prompt'] = loc_image_prompt
        info_dict['loc_image_ground_truth'] = loc_image_ground_truth
        info_dict['loc_image_predict'] = loc_image_predict
        del loc_image_prompt,loc_image_ground_truth,loc_image_predict
        torch.cuda.empty_cache()
        


        if "T_Rel_1" in batch:
            info_dict['T_Rel_1_prompt'] = T_Rel_1_prompt
            info_dict['T_Rel_1_ground_truth'] = T_Rel_1_ground_truth
            info_dict['T_Rel_1_predict'] = T_Rel_1_predict
            del T_Rel_1_prompt,T_Rel_1_ground_truth,T_Rel_1_predict
            torch.cuda.empty_cache()
            
            info_dict['T_Rel_2_prompt'] = T_Rel_2_prompt
            info_dict['T_Rel_2_ground_truth'] = T_Rel_2_ground_truth
            info_dict['T_Rel_2_predict'] = T_Rel_2_predict
            del T_Rel_2_prompt,T_Rel_2_ground_truth,T_Rel_2_predict
            torch.cuda.empty_cache()
            
            info_dict['M_Rel_1_prompt'] = M_Rel_1_prompt
            info_dict['M_Rel_1_ground_truth'] = M_Rel_1_ground_truth
            info_dict['M_Rel_1_predict'] = M_Rel_1_predict
            del M_Rel_1_prompt,M_Rel_1_ground_truth,M_Rel_1_predict
            torch.cuda.empty_cache()
            
            info_dict['M_Rel_2_prompt'] = M_Rel_2_prompt
            info_dict['M_Rel_2_ground_truth'] = M_Rel_2_ground_truth
            info_dict['M_Rel_2_predict'] = M_Rel_2_predict
            del M_Rel_2_prompt,M_Rel_2_ground_truth,M_Rel_2_predict
            torch.cuda.empty_cache()
            
            info_dict['Gen_M_Rel_1_prompt'] = Gen_M_Rel_1_prompt
            info_dict['Gen_M_Rel_1_ground_truth'] = Gen_M_Rel_1_ground_truth
            info_dict['Gen_M_Rel_1_predict'] = Gen_M_Rel_1_predict
            del Gen_M_Rel_1_prompt,Gen_M_Rel_1_ground_truth,Gen_M_Rel_1_predict
            torch.cuda.empty_cache()
            
            info_dict['Gen_M_Rel_2_prompt'] = Gen_M_Rel_2_prompt
            info_dict['Gen_M_Rel_2_ground_truth'] = Gen_M_Rel_2_ground_truth
            info_dict['Gen_M_Rel_2_predict'] = Gen_M_Rel_2_predict
            del Gen_M_Rel_2_prompt,Gen_M_Rel_2_ground_truth,Gen_M_Rel_2_predict
            torch.cuda.empty_cache()
        else:
            
            info_dict['T_Rel_prompt'] = T_Rel_prompt
            info_dict['T_Rel_ground_truth'] = T_Rel_ground_truth
            info_dict['T_Rel_predict'] = T_Rel_predict
            del T_Rel_prompt,T_Rel_ground_truth,T_Rel_predict
            torch.cuda.empty_cache()
            
            info_dict['M_Rel_prompt'] = M_Rel_prompt
            info_dict['M_Rel_ground_truth'] = M_Rel_ground_truth
            info_dict['M_Rel_predict'] = M_Rel_predict
            del M_Rel_prompt,M_Rel_ground_truth,M_Rel_predict
            torch.cuda.empty_cache()
            
            info_dict['Gen_M_Rel_prompt'] = Gen_M_Rel_prompt
            info_dict['Gen_M_Rel_ground_truth'] = Gen_M_Rel_ground_truth
            info_dict['Gen_M_Rel_predict'] = Gen_M_Rel_predict
            del Gen_M_Rel_prompt,Gen_M_Rel_ground_truth,Gen_M_Rel_predict
            torch.cuda.empty_cache()

        info_dict = {**info_dict, **model_info}

        return l_total, l_edit , l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        outer_acc = f"{stats['edit/acc_val']:<12.5f}"
        image_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"
        ##########################################################################
        if stats['knowledge_type_val'] == 1 or stats['knowledge_type_val'] == 0:
        # if stats['T_Rel_1/acc_val'] is not None:
            
            T_Rel_1_acc = f"{stats['T_Rel_1/acc_val']:<12.5f}"
            T_Rel_2_acc = f"{stats['T_Rel_2/acc_val']:<12.5f}"
            T_Rel_average_acc = f"{stats['T_Rel_average/acc_val']:<12.5f}"
           
            M_Rel_1_acc = f"{stats['M_Rel_1/acc_val']:<12.5f}"
            M_Rel_2_acc = f"{stats['M_Rel_2/acc_val']:<12.5f}"
            M_Rel_average_acc = f"{stats['M_Rel_average/acc_val']:<12.5f}"
            
            Gen_M_Rel_1_acc = f"{stats['Gen_M_Rel_1/acc_val']:<12.5f}"
            Gen_M_Rel_2_acc = f"{stats['Gen_M_Rel_2/acc_val']:<12.5f}"
            Gen_M_Rel_average_acc = f"{stats['Gen_M_Rel_average/acc_val']:<12.5f}"

            LOG.info(
          f"Step {prog}  image_acc: {image_acc} inner_acc: {inner_acc} outer_acc: {outer_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc} T_Rel_1_acc:{T_Rel_1_acc}  T_Rel_2_acc:{T_Rel_2_acc} T_Rel_average_acc:{T_Rel_average_acc}  M_Rel_1_acc:{M_Rel_1_acc} M_Rel_2_acc:{M_Rel_2_acc}  M_Rel_average_acc:{M_Rel_average_acc}   Gen_M_Rel_1_acc:{Gen_M_Rel_1_acc}  Gen_M_Rel_2_acc:{Gen_M_Rel_2_acc}  Gen_M_Rel_average_acc:{Gen_M_Rel_average_acc}"
        )

        elif stats['knowledge_type_val'] == 2:
        # else:
            
            T_Rel_acc = f"{stats['T_Rel/acc_val']:<12.5f}"
            M_Rel_acc = f"{stats['M_Rel/acc_val']:<12.5f}"
            Gen_M_Rel_acc = f"{stats['Gen_M_Rel/acc_val']:<12.5f}"
            
            LOG.info(
            f"Step {prog}  image_acc: {image_acc} inner_acc: {inner_acc} outer_acc: {outer_acc} it_time: {elapsed:.4f} loc_acc: {loc_acc}, image_loc: {loc_image_acc} T_Rel_acc:{T_Rel_acc}  M_Rel_acc:{M_Rel_acc}  Gen_M_Rel_acc:{Gen_M_Rel_acc} "
        )
        ##########################################################################
        if 'port/acc_val' in stats:
            LOG.info(f"step {prog} port_acc: {stats['port/acc_val']:<12.5f}")

    def validate(self, steps=None, log: bool = False, json_output_file: str = "validation_results.json"):   
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        from datetime import datetime
        cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        

        self.config.results_dir = f"./results/{self.config.data_type}"

        # blip2 inner_params:- Qformer 
        # llava inner_params:- mm_projector
        # minigpt4 inner_params:- Qformer 
        # qwenvl inner_params:- transformer.visual.transformer.resblocks.47.mlp.c_fc.weight - transformer.visual.transformer.resblocks.47.mlp.c_fc.bias
        # owl2 inner_params:- model.vision_model.encoder.layers.23.mlp.fc1.weight   - model.vision_model.encoder.layers.23.mlp.fc1.bias

        if any(substring in self.config.inner_params[0] for substring in ["Qformer", "mm_projector", "transformer.visual.transformer.resblocks.47.mlp.c_fc.weight", "model.vision_model.encoder.layers.23.mlp.fc1.weight"]) and self.config.alg == 'ft':
            self.config.alg == 'ft-q'


        model_dir = os.path.join(self.config.results_dir, "models", self.config.alg)
        
        json_output_file = f"{model_dir}/{cur_time}_{self.config.model_name}_{self.config.alg}_decode_output_results.json"

        start_time = time.time()

        
        with open(json_output_file, 'w') as f:
            f.write('[\n')

        for val_step, batch in tqdm(enumerate(self.val_loader), total=steps, desc="Validation", ncols=100):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

           
            for key, value in info_dict.items():
                if isinstance(value, torch.Tensor):
                    info_dict[key] = value.tolist()

            
            with open(json_output_file, 'a') as f:
                json.dump(info_dict, f, indent=4)  
                if val_step < steps - 1:
                    f.write(",\n")  
                else:
                    f.write("\n")  

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        
        with open(json_output_file, 'a') as f:
            f.write(']\n')

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats
