"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm
from copy import deepcopy

from ..trainer.mPLUG_Owl2.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from ..trainer.mPLUG_Owl2.mplug_owl2.mm_utils import tokenizer_image_token, process_images

class CaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, no_image=False, hop=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLaVA":
            # vis_processor = transformers.CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
            vis_processor = transformers.CLIPImageProcessor.from_pretrained("/scratch2/nlp/plm/clip-vit-large-patch14-336")
        elif config.model_class ==  "qwen-vl":
            vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
        elif "owl-2" in config.model_name.lower():
            from transformers.models.clip.image_processing_clip import CLIPImageProcessor
            vis_processor = CLIPImageProcessor.from_pretrained(config.name, trust_remote_code=True)
        else:
            raise NotImplementedError("unknown model class")

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
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: "

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]
        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop should be 1, 2, 3, or 4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]
        for record in tqdm(self.annotation, ncols=120, desc='Loading Data'):
            
            if record['alt'] == "":
                continue
            if hop and 'port_new' not in record.keys():
                continue
            
            image_path = os.path.join(self.vis_root, record["image"])
            rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            locality_image_path = os.path.join(self.vis_root, record['m_loc'])
            

            if record['knowledge_type'] == 'visual_knowledge' :
                one_hop_img_path = os.path.join(self.vis_root, record['one_hop_img'])

                item = {
                    'prompt': record['src'],
                    'pred': record['pred'],
                    'target': record['alt'],
                    'rephrase_prompt': record['rephrase'],
                    'image': image_path,
                    'image_rephrase': rephrase_image_path,
                    'one_hop_img': one_hop_img_path,
                    'image_rephrase_question': record['image_rephrase_question'],
                    'knowledge_type': record['knowledge_type'],
                    'cond': "{} >> {} || {}".format(
                        record['pred'],
                        record['alt'],
                        record['src']
                    )
                }
            elif record['knowledge_type'] == 'user_specific':
                one_hop_img_path = os.path.join(self.vis_root, record['one_hop_img'])
                item = {
                    'prompt': record['src'],
                    'pred': record['pred'],
                    'target': record['alt'],
                    'rephrase_prompt': record['rephrase'],
                    'image': image_path,
                    'image_rephrase': rephrase_image_path,
                    'one_hop_img': one_hop_img_path,
                    'knowledge_type': record['knowledge_type'],
                    'cond': "{} >> {} || {}".format(
                        record['pred'],
                        record['alt'],
                        record['src']
                    )
                }
            else:
                item = {
                    'prompt': record['src'],
                    'pred': record['pred'],
                    'target': record['alt'],
                    'rephrase_prompt': record['rephrase'],
                    'image': image_path,
                    'image_rephrase': rephrase_image_path,
                    'knowledge_type': record['knowledge_type'],
                    'cond': "{} >> {} || {}".format(
                        record['pred'],
                        record['alt'],
                        record['src']
                    )
                }   

  
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            
            item['multimodal_locality_image'] = locality_image_path
            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']

            
            ##########################################################################################################
            
            #text-reliability
            if  item['knowledge_type']=='entity_level' or item['knowledge_type']=='user_specific':
                item['rel_prompt_1'] = record['rel_1']
                item['rel_ground_truth_1'] = record['rel_ans_1'] if 'rel_ans_1' in record  else record['rel_1_ans']

                item['rel_prompt_2'] = record['rel_2']
                item['rel_ground_truth_2'] = record['rel_ans_2'] if 'rel_ans_2' in record  else record['rel_2_ans']
                #multimodal-reliability
                item['m_rel_prompt_1'] = record['m_rel_1']
                item['m_rel_ground_truth_1'] = record['m_rel_ans_1'] if 'm_rel_ans_1' in record  else record['m_rel_1_ans']

                item['m_rel_prompt_2'] = record['m_rel_2']
                item['m_rel_ground_truth_2'] = record['m_rel_ans_2'] if 'm_rel_ans_2' in record  else record['m_rel_2_ans']
                if item['knowledge_type']=='entity_level':
                    item['knowledge_type'] = 0
                elif item['knowledge_type']=='user_specific':
                    item['knowledge_type'] = 1
                
         
            elif 'visual_knowledge' in item['knowledge_type']:
                
                item['rel_prompt'] = record['rel']
                item['rel_ground_truth'] = record['rel_ans']

                #multimodal-reliability
                item['m_rel_prompt'] = record['m_rel']
                item['m_rel_ground_truth'] = record['m_rel_ans']
                #image_rephrase_question

                item['knowledge_type'] = 2
            ##########################################################################################################


            if hop and 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question']
                        port_a = ports['Q&A']['Answer']
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break
                
                if not find_hop:
                    continue
            data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data
        self.no_image = no_image

    def __getitem__(self, index):
        if self.no_image:
            return self._data[index]

        data = deepcopy(self._data[index])        
        # load image
        image_path = data['image']
        rephrase_image_path = data['image_rephrase']
        locality_image_path = data['multimodal_locality_image']
        
        image = Image.open(image_path).convert("RGB")
        rephrase_image = Image.open(rephrase_image_path).convert("RGB")
        locality_image = Image.open(locality_image_path).convert("RGB")
        ###############################
        if 'one_hop_img' in data:
            one_hop_img_path = data['one_hop_img']
            one_hop_img = Image.open(one_hop_img_path).convert("RGB")
        #############################
        
        if self.config.model_class == "Blip2OPT":
            image = self.vis_processor(image)
            rephrase_image = self.vis_processor(rephrase_image)
            locality_image = self.vis_processor(locality_image)

            if 'one_hop_img' in data and one_hop_img is not None:
                one_hop_img = self.vis_processor(one_hop_img)

        elif self.config.model_class == "LLaVA":
            image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
            if 'one_hop_img' in data and one_hop_img is not None:
                one_hop_img = self.vis_processor(one_hop_img, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
        elif self.config.model_class == "qwen-vl":
            # image = os.path.join(self.vis_root, image_path)
            # rephrase_image = os.path.join(self.rephrase_root, rephrase_image_path)
            # locality_image = os.path.join(self.vis_root, locality_image_path)
            # if'one_hop_img' in data and one_hop_img is not None:
            #     one_hop_img = os.path.join(self.vis_root, one_hop_img_path)

            image = os.path.join(image_path)
            rephrase_image = os.path.join(rephrase_image_path)
            locality_image = os.path.join(locality_image_path)
            if'one_hop_img' in data and one_hop_img is not None:
                one_hop_img = os.path.join(one_hop_img_path)


        elif self.config.model_name == "owl-2":
            
                    
            _image = Image.open(image_path).convert('RGB')
            max_edge = max(_image.size) 
            image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)

            _image = Image.open(rephrase_image_path).convert('RGB')
            max_edge = max(_image.size) 
            rephrase_image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)

            _image = Image.open(locality_image_path).convert('RGB')
            max_edge = max(_image.size) 
            locality_image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)

            if 'one_hop_img' in data and one_hop_img is not None:
                _image = Image.open(one_hop_img_path).convert('RGB')
                max_edge = max(_image.size) 
                one_hop_img = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)


        else:
            raise NotImplementedError

        data['image'] = image
        data['image_rephrase'] = rephrase_image
        data['multimodal_locality_image'] = locality_image
        if 'one_hop_img' in data and one_hop_img is not None:
            data['one_hop_img'] = one_hop_img
        return data
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch] if "owl-2" not in self.config.model_name else [b['image'] for b in batch][0]
        image_rephrase = [b['image_rephrase'] for b in batch] if "owl-2" not in self.config.model_name else [b['image_rephrase'] for b in batch][0]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch] if "owl-2" not in self.config.model_name else [b['multimodal_locality_image'] for b in batch][0]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        tokenizer = AutoTokenizer.from_pretrained(self.config.name, use_fast=False) if self.config.model_name == "owl-2" else None

        if 'one_hop_img' in batch[0] and batch[0]['knowledge_type'] == 1:
            #text-reliability
            one_hop_img = [b['one_hop_img'] for b in batch] if "owl-2" not in self.config.model_name else [b['one_hop_img'] for b in batch][0]
            rel_q_1 = [b['rel_prompt_1'] for b in batch]
            rel_a_1 = [b['rel_ground_truth_1'] for b in batch]
            rel_q_2 = [b['rel_prompt_2'] for b in batch]
            rel_a_2 = [b['rel_ground_truth_2'] for b in batch]
            #multimodal-reliability
            m_rel_q_1 = [b['m_rel_prompt_1'] for b in batch]
            m_rel_a_1 = [b['m_rel_ground_truth_1'] for b in batch]
            m_rel_q_2 = [b['m_rel_prompt_2'] for b in batch]
            m_rel_a_2 = [b['m_rel_ground_truth_2'] for b in batch]  
        
        
        elif 'one_hop_img' in batch[0] and batch[0]['knowledge_type'] == 2:  
            one_hop_img = [b['one_hop_img'] for b in batch] if "owl-2" not in self.config.model_name else [b['one_hop_img'] for b in batch][0]
            #text-reliability
            rel_q = [b['rel_prompt'] for b in batch]
            rel_a = [b['rel_ground_truth'] for b in batch]
            #multimodal-reliability
            m_rel_q = [b['m_rel_prompt'] for b in batch]
            m_rel_a = [b['m_rel_ground_truth'] for b in batch]
            image_rephrase_q = [b['image_rephrase_question'] for b in batch]

        else:
            #text-reliability
            rel_q_1 = [b['rel_prompt_1'] for b in batch]
            rel_a_1 = [b['rel_ground_truth_1'] for b in batch]
            rel_q_2 = [b['rel_prompt_2'] for b in batch]
            rel_a_2 = [b['rel_ground_truth_2'] for b in batch]
            #multimodal-reliability
            m_rel_q_1 = [b['m_rel_prompt_1'] for b in batch]
            m_rel_a_1 = [b['m_rel_ground_truth_1'] for b in batch]
            m_rel_q_2 = [b['m_rel_prompt_2'] for b in batch]
            m_rel_a_2 = [b['m_rel_ground_truth_2'] for b in batch]
                 
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{src[0]} {trg[0]}', return_tensors='pt')["input_ids"]
        edit_inner['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + src[0] + " " + trg[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        edit_inner['prompt'] = src
        edit_inner['ground_truth'] = trg
        edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # edit_outer
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{rephrase[0]} {trg[0]}', return_tensors='pt')["input_ids"]
        edit_outer['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + rephrase[0] + " " + trg[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        edit_outer['prompt'] = rephrase
        edit_outer['ground_truth'] = trg
        edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
        edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image_rephrase
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['inputs'] = self.tok(f'Picture 1: <img>{image_rephrase[0]}</img>\n{src[0]} {trg[0]}', return_tensors='pt')["input_ids"]
        edit_outer_image['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + src[0] + " " + trg[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        edit_outer_image['prompt'] = src
        edit_outer_image['ground_truth'] = trg
        edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
        edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        
        # loc
        loc = {}
        loc['image'] = torch.zeros(1, 3, 448, 448) if "owl-2" in self.config.model_name else None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['inputs'] = self.tok(f"{loc_q[0]} {loc_a[0]}", return_tensors='pt')["input_ids"]
        loc['input_ids'] = tokenizer_image_token(loc_q[0] + " " + loc_a[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        loc['prompt'] = loc_q
        loc['ground_truth'] = loc_a
        loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # m_loc
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else m_loc_image
        loc_image['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['inputs'] = self.tok(f'Picture 1: <img>{m_loc_image[0]}</img>\n{m_loc_q[0]} {m_loc_a[0]}', return_tensors='pt')["input_ids"]
        loc_image['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + m_loc_q[0] + " " + m_loc_a[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
        loc_image['prompt'] = m_loc_q
        loc_image['ground_truth'] = m_loc_a
        loc_image['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
        loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        ##########################################################################################################
        if batch[0]['knowledge_type']==0 or batch[0]['knowledge_type']==1:
            knowledge_type = {}
            if batch[0]['knowledge_type']==0:
                knowledge_type['knowledge_type'] =0
            elif batch[0]['knowledge_type']==1:
                knowledge_type['knowledge_type'] =1
            
            T_Rel_1 = {}
            T_Rel_1['image'] = torch.zeros(1, 3, 448, 448) if "owl-2" in self.config.model_name else None
            T_Rel_1['text_input'] = [" ".join([q, a]) for q, a in zip(rel_q_1, rel_a_1)]
            ##############################################################################
            T_Rel_1['inputs'] = self.tok(f"{rel_q_1[0]} {rel_a_1[0]}", return_tensors='pt')["input_ids"]
            T_Rel_1['input_ids'] = tokenizer_image_token(rel_q_1[0] + " " + rel_a_1[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
            ##############################################################################
            T_Rel_1['prompt'] = rel_q_1
            T_Rel_1['ground_truth'] = rel_a_1
            T_Rel_1['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in rel_q_1]
            T_Rel_1['labels'] = self.tok(rel_a_1, add_special_tokens=False, return_tensors="pt",)["input_ids"]

            T_Rel_2 = {}
            T_Rel_2['image'] = torch.zeros(1, 3, 448, 448) if "owl-2" in self.config.model_name else None
            T_Rel_2['text_input'] = [" ".join([q, a]) for q, a in zip(rel_q_2, rel_a_2)]
            ##############################################################################
            T_Rel_2['inputs'] = self.tok(f"{rel_q_2[0]} {rel_a_2[0]}", return_tensors='pt')["input_ids"]
            T_Rel_2['input_ids'] = tokenizer_image_token(rel_q_2[0] + " " + rel_a_2[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
            ##############################################################################
            T_Rel_2['prompt'] = rel_q_2
            T_Rel_2['ground_truth'] = rel_a_2
            T_Rel_2['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in rel_q_2]
            T_Rel_2['labels'] = self.tok(rel_a_2, add_special_tokens=False, return_tensors="pt",)["input_ids"]

           
            M_Rel_1 = {}
            M_Rel_1['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
            M_Rel_1['text_input'] = [" ".join([q, a]) for q, a in zip(m_rel_q_1, m_rel_a_1)]
            ##############################################################################
            M_Rel_1['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{m_rel_q_1[0]} {m_rel_a_1[0]}', return_tensors='pt')["input_ids"]
            M_Rel_1['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + m_rel_q_1[0] + " " + m_rel_a_1[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
            ##############################################################################
            # M_Rel_1['labels'] = m_rel_a_1
            M_Rel_1['prompt'] = m_rel_q_1
            M_Rel_1['ground_truth'] = m_rel_a_1
            M_Rel_1['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_rel_q_1]
            M_Rel_1['labels'] = self.tok(m_rel_a_1, add_special_tokens=False, return_tensors="pt",)["input_ids"]

            M_Rel_2 = {}
            M_Rel_2['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
            M_Rel_2['text_input'] = [" ".join([q, a]) for q, a in zip(m_rel_q_2, m_rel_a_2)]
            ##############################################################################
            M_Rel_2['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{m_rel_q_2[0]} {m_rel_a_2[0]}', return_tensors='pt')["input_ids"]
            M_Rel_2['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + m_rel_q_2[0] + " " + m_rel_a_2[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
            ##############################################################################
            # M_Rel_2['labels'] = m_rel_a_2
            M_Rel_2['prompt'] = m_rel_q_2
            M_Rel_2['ground_truth'] = m_rel_a_2
            M_Rel_2['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_rel_q_2]
            M_Rel_2['labels'] = self.tok(m_rel_a_2, add_special_tokens=False, return_tensors="pt",)["input_ids"]

            
            Gen_M_Rel_1 = {}
            Gen_M_Rel_1['image'] = torch.stack(image_rephrase, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image_rephrase
            Gen_M_Rel_1['text_input'] = [" ".join([q, a]) for q, a in zip(m_rel_q_1, m_rel_a_1)]
            ##############################################################################
            Gen_M_Rel_1['inputs'] = self.tok(f'Picture 1: <img>{image_rephrase[0]}</img>\n{m_rel_q_1[0]} {m_rel_a_1[0]}', return_tensors='pt')["input_ids"]
            Gen_M_Rel_1['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + m_rel_q_1[0] + " " + m_rel_a_1[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
            ##############################################################################
            # Gen_M_Rel_1['labels'] = m_rel_a_1
            Gen_M_Rel_1['prompt'] = m_rel_q_1
            Gen_M_Rel_1['ground_truth'] = m_rel_a_1
            Gen_M_Rel_1['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_rel_q_1]
            Gen_M_Rel_1['labels'] = self.tok(m_rel_a_1, add_special_tokens=False, return_tensors="pt",)["input_ids"]

            Gen_M_Rel_2 = {}
            Gen_M_Rel_2['image'] = torch.stack(image_rephrase, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image_rephrase
            Gen_M_Rel_2['text_input'] = [" ".join([q, a]) for q, a in zip(m_rel_q_2, m_rel_a_2)]
            ##############################################################################
            Gen_M_Rel_2['inputs'] = self.tok(f'Picture 1: <img>{image_rephrase[0]}</img>\n{m_rel_q_2[0]} {m_rel_a_2[0]}', return_tensors='pt')["input_ids"]
            Gen_M_Rel_2['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + m_rel_q_2[0] + " " + m_rel_a_2[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
            ##############################################################################
            # Gen_M_Rel_2['labels'] = m_rel_a_2
            Gen_M_Rel_2['prompt'] = m_rel_q_2
            Gen_M_Rel_2['ground_truth'] = m_rel_a_2
            Gen_M_Rel_2['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_rel_q_2]
            Gen_M_Rel_2['labels'] = self.tok(m_rel_a_2, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        elif batch[0]['knowledge_type']== 2:
            knowledge_type = {}
            knowledge_type['knowledge_type'] =2

            
            T_Rel = {}
            T_Rel['image'] = torch.zeros(1, 3, 448, 448) if "owl-2" in self.config.model_name else None
            T_Rel['text_input'] = [" ".join([q, a]) for q, a in zip(rel_q, rel_a)]
            ##############################################################################
            T_Rel['inputs'] = self.tok(f"{rel_q[0]} {rel_a[0]}", return_tensors='pt')["input_ids"]
            T_Rel['input_ids'] = tokenizer_image_token(rel_q[0] + " " + rel_a[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
            ##############################################################################
            # T_Rel['labels'] = rel_a
            T_Rel['prompt'] = rel_q
            T_Rel['ground_truth'] = rel_a
            T_Rel['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in rel_q]
            T_Rel['labels'] = self.tok(rel_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]

            
            M_Rel = {}
            M_Rel['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
            M_Rel['text_input'] = [" ".join([q, a]) for q, a in zip(m_rel_q, m_rel_a)]
            ##############################################################################
            M_Rel['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{m_rel_q[0]} {m_rel_a[0]}', return_tensors='pt')["input_ids"]
            M_Rel['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + m_rel_q[0] + " " + m_rel_a[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
            ##############################################################################
            # M_Rel['labels'] = m_rel_a
            M_Rel['prompt'] = m_rel_q
            M_Rel['ground_truth'] = m_rel_a
            M_Rel['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_rel_q]
            M_Rel['labels'] = self.tok(m_rel_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]

            
            Gen_M_Rel = {}
            Gen_M_Rel['image'] = torch.stack(image_rephrase, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image_rephrase
            Gen_M_Rel['text_input'] = [" ".join([q, a]) for q, a in zip(image_rephrase_q, m_rel_a)]
            ##############################################################################
            Gen_M_Rel['inputs'] = self.tok(f'Picture 1: <img>{image_rephrase[0]}</img>\n{image_rephrase_q[0]} {m_rel_a[0]}', return_tensors='pt')["input_ids"]
            Gen_M_Rel['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + image_rephrase_q[0] + " " + m_rel_a[0], tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
            ##############################################################################
            # Gen_M_Rel['labels'] = m_rel_a
            Gen_M_Rel['prompt'] = image_rephrase_q
            Gen_M_Rel['ground_truth'] = m_rel_a
            Gen_M_Rel['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in image_rephrase_q]
            Gen_M_Rel['labels'] = self.tok(m_rel_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]


        ##########################################################################################################


        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)

        edit_ports = None


        if 'portability_prompt' in batch[0].keys():
            edit_ports = []
            for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                port = {}
                if batch[0]['knowledge_type']==0:
                    port['image'] = torch.stack(image, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else image
                    port['inputs'] = self.tok(f'Picture 1: <img>{image[0]}</img>\n{port_q[0]} {port_a[0]}', return_tensors='pt')["input_ids"]
                elif batch[0]['knowledge_type']==1 or batch[0]['knowledge_type']== 2:
                    port['image'] = torch.stack(one_hop_img, dim=0) if ("qwen-vl" not in self.config.model_name and "owl-2" not in self.config.model_name) else one_hop_img
                    port['inputs'] = self.tok(f'Picture 1: <img>{one_hop_img[0]}</img>\n{port_q[0]} {port_a[0]}', return_tensors='pt')["input_ids"]
                port['text_input'] = [' '.join([port_q, port_a])]
                port['input_ids'] = tokenizer_image_token(DEFAULT_IMAGE_TOKEN + port_q + " " + port_a, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0) if self.config.model_name == "owl-2" else None
                port['prompt'] = port_q
                port['ground_truth'] = port_a
                port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
                port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt",)["input_ids"]
                edit_ports.append(port)


        ##################################################################################################
        if batch[0]['knowledge_type']==0 or batch[0]['knowledge_type']==1:
            batch = {
                "edit_inner": edit_inner,   # multimodal summary reliability（origin image）
                "edit_outer": edit_outer,
                "edit_outer_image": edit_outer_image,  # multimodal summary reliability（image_rephrase）
                "loc": loc,    # T-Loc
                "loc_image": loc_image,   # I-Loc
                "T_Rel_1": T_Rel_1,      # T-Rel
                "T_Rel_2": T_Rel_2,      # T-Rel
                "M_Rel_1": M_Rel_1,      # I-Rel
                "M_Rel_2": M_Rel_2,      # I-Rel
                "Gen_M_Rel_1": Gen_M_Rel_1,   # I-Gen
                "Gen_M_Rel_2": Gen_M_Rel_2,   # I-Gen
                'port': edit_ports,
                'knowledge_type':knowledge_type,
                "cond": cond
            }
        elif batch[0]['knowledge_type']== 2:
            batch = {
                "edit_inner": edit_inner,   # multimodal summary reliability（origin image）
                "edit_outer": edit_outer,
                "edit_outer_image": edit_outer_image,  # multimodal summary reliability（image_rephrase）
                "loc": loc,    # T-Loc
                "loc_image": loc_image,   # I-Loc
                "T_Rel": T_Rel,      # T-Rel
                "M_Rel": M_Rel,      # I-Rel
                "Gen_M_Rel": Gen_M_Rel,   # I-Gen
                'port': edit_ports,
                'knowledge_type':knowledge_type,
                "cond": cond
            }
        ##################################################################################################
            
        return dict_to(batch, self.config.device)
