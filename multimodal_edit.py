import os
import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, \
    MENDMultimodalHparams \
    , SERACMultimodalHparams, FTMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import sys
import argparse


def VLKEB_print_result(metrics, save_path=None):
    rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
    rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
    rephrase_image_acc = mean([m['post']['rephrase_image_acc'].item() for m in metrics])
    locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
    locality_image_acc = mean([m['post']['locality_image_acc'].item() for m in metrics])
    print(f'rewrite_acc: {rewrite_acc}')
    print(f'rephrase_acc: {rephrase_acc}')
    print(f'rephrase_image_acc: {rephrase_image_acc}')
    print(f'locality_acc: {locality_acc}')
    print(f'locality_image_acc: {locality_image_acc}')

    ### portability
    if 'portability_acc' in metrics[0]['post']:
        portability_acc = mean([m['post']['portability_acc'].item() for m in metrics])
        print(f'portability_acc: {portability_acc}')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(f'rewrite_acc: {rewrite_acc}\n')
            f.write(f'rephrase_acc: {rephrase_acc}\n')
            f.write(f'rephrase_image_acc: {rephrase_image_acc}\n')
            f.write(f'locality_acc: {locality_acc}\n')
            f.write(f'locality_image_acc: {locality_image_acc}\n')

            #### portability
            if 'portability_acc' in metrics[0]['post']:
                f.write(f'portability_acc: {portability_acc}\n')


###########################################################################
def MMKE_print_result(metrics, save_path=None):
    if metrics[0]['post']['knowledge_type'] == 0 or metrics[0]['post']['knowledge_type'] == 1:
        memory_alloc_max = mean([m['post']['memory_alloc_max'] for m in metrics])
        memory_res_max = mean([m['post']['memory_res_max'] for m in metrics])
        rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
        rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
        rephrase_image_acc = mean([m['post']['rephrase_image_acc'].item() for m in metrics])
        locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
        locality_image_acc = mean([m['post']['locality_image_acc'].item() for m in metrics])
        
        rel_prompt_1_acc = mean([m['post']['rel_prompt_1_acc'].item() for m in metrics])
        rel_prompt_2_acc = mean([m['post']['rel_prompt_2_acc'].item() for m in metrics])
        rel_prompt_acc_average = (rel_prompt_1_acc + rel_prompt_2_acc) / 2

        m_rel_prompt_1_image_acc = mean([m['post']['m_rel_prompt_1_image_acc'].item() for m in metrics])
        m_rel_prompt_2_image_acc = mean([m['post']['m_rel_prompt_2_image_acc'].item() for m in metrics])
        m_rel_prompt_image_acc_average = (m_rel_prompt_1_image_acc + m_rel_prompt_2_image_acc) / 2

        m_rel_prompt_1_image_rephrase_acc = mean(
            [m['post']['m_rel_prompt_1_image_rephrase_acc'].item() for m in metrics])
        m_rel_prompt_2_image_rephrase_acc = mean(
            [m['post']['m_rel_prompt_2_image_rephrase_acc'].item() for m in metrics])
        m_rel_prompt_image_rephrase_acc_average = (
                                                              m_rel_prompt_1_image_rephrase_acc + m_rel_prompt_2_image_rephrase_acc) / 2

        print(f'memory_alloc_max: {memory_alloc_max}')
        print(f'memory_res_max: {memory_res_max}')

        print(f'rewrite_acc: {rewrite_acc}')
        print(f'rephrase_acc: {rephrase_acc}')
        print(f'rephrase_image_acc: {rephrase_image_acc}')
        print(f'locality_acc: {locality_acc}')
        print(f'locality_image_acc: {locality_image_acc}')
        
        print(f'rel_prompt_1_acc: {rel_prompt_1_acc}')
        print(f'rel_prompt_2_acc: {rel_prompt_2_acc}')
        print(f'rel_prompt_acc_average: {rel_prompt_acc_average}')

        print(f'm_rel_prompt_1_image_acc: {m_rel_prompt_1_image_acc}')
        print(f'm_rel_prompt_2_image_acc: {m_rel_prompt_2_image_acc}')
        print(f'm_rel_prompt_image_acc_average: {m_rel_prompt_image_acc_average}')

        print(f'm_rel_prompt_1_image_rephrase_acc: {m_rel_prompt_1_image_rephrase_acc}')
        print(f'm_rel_prompt_2_image_rephrase_acc: {m_rel_prompt_2_image_rephrase_acc}')
        print(f'm_rel_prompt_image_rephrase_acc_average: {m_rel_prompt_image_rephrase_acc_average}')

        ### portability
        if 'portability_acc' in metrics[0]['post']:
            portability_acc = mean([m['post']['portability_acc'].item() for m in metrics])
            print(f'portability_acc: {portability_acc}')

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:

                f.write(f'memory_alloc_max: {memory_alloc_max}\n')
                f.write(f'memory_res_max: {memory_res_max}\n')

                f.write(f'rewrite_acc: {rewrite_acc}\n')
                f.write(f'rephrase_acc: {rephrase_acc}\n')
                f.write(f'rephrase_image_acc: {rephrase_image_acc}\n')
                f.write(f'locality_acc: {locality_acc}\n')
                f.write(f'locality_image_acc: {locality_image_acc}\n')
                
                f.write(f'rel_prompt_1_acc: {rel_prompt_1_acc}\n')
                f.write(f'rel_prompt_2_acc: {rel_prompt_2_acc}\n')
                f.write(f'rel_prompt_acc_average: {rel_prompt_acc_average}\n')

                f.write(f'm_rel_prompt_1_image_acc: {m_rel_prompt_1_image_acc}\n')
                f.write(f'm_rel_prompt_2_image_acc: {m_rel_prompt_2_image_acc}\n')
                f.write(f'm_rel_prompt_image_acc_average: {m_rel_prompt_image_acc_average}\n')

                f.write(f'm_rel_prompt_1_image_rephrase_acc: {m_rel_prompt_1_image_rephrase_acc}\n')
                f.write(f'm_rel_prompt_2_image_rephrase_acc: {m_rel_prompt_2_image_rephrase_acc}\n')
                f.write(f'm_rel_prompt_image_rephrase_acc_average: {m_rel_prompt_image_rephrase_acc_average}\n')

                #### portability
                if 'portability_acc' in metrics[0]['post']:
                    f.write(f'portability_acc: {portability_acc}\n')

    elif metrics[0]['post']['knowledge_type'] == 2:
        memory_alloc_max = mean([m['post']['memory_alloc_max'] for m in metrics])
        memory_res_max = mean([m['post']['memory_res_max'] for m in metrics])
        rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
        rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
        rephrase_image_acc = mean([m['post']['rephrase_image_acc'].item() for m in metrics])
        locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
        locality_image_acc = mean([m['post']['locality_image_acc'].item() for m in metrics])
        
        rel_prompt_acc = mean([m['post']['rel_prompt_acc'].item() for m in metrics])

        m_rel_prompt_image_acc = mean([m['post']['m_rel_prompt_image_acc'].item() for m in metrics])

        m_rel_prompt_image_rephrase_acc = mean([m['post']['m_rel_prompt_image_rephrase_acc'].item() for m in metrics])

        print(f'memory_alloc_max: {memory_alloc_max}')
        print(f'memory_res_max: {memory_res_max}')
        print(f'rewrite_acc: {rewrite_acc}')
        print(f'rephrase_acc: {rephrase_acc}')
        print(f'rephrase_image_acc: {rephrase_image_acc}')
        print(f'locality_acc: {locality_acc}')
        print(f'locality_image_acc: {locality_image_acc}')
        
        print(f'rel_prompt_acc: {rel_prompt_acc}')
        print(f'm_rel_prompt_image_acc: {m_rel_prompt_image_acc}')
        print(f'm_rel_prompt_image_rephrase_acc: {m_rel_prompt_image_rephrase_acc}')

        ### portability
        if 'portability_acc' in metrics[0]['post']:
            portability_acc = mean([m['post']['portability_acc'].item() for m in metrics])
            print(f'portability_acc: {portability_acc}')

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(f'memory_alloc_max: {memory_alloc_max}\n')
                f.write(f'memory_res_max: {memory_res_max}\n')
                f.write(f'rewrite_acc: {rewrite_acc}\n')
                f.write(f'rephrase_acc: {rephrase_acc}\n')
                f.write(f'rephrase_image_acc: {rephrase_image_acc}\n')
                f.write(f'locality_acc: {locality_acc}\n')
                f.write(f'locality_image_acc: {locality_image_acc}\n')
                
                f.write(f'rel_prompt_acc: {rel_prompt_acc}\n')
                f.write(f'm_rel_prompt_image_acc: {m_rel_prompt_image_acc}\n')
                f.write(f'm_rel_prompt_image_rephrase_acc: {m_rel_prompt_image_rephrase_acc}\n')
                #### portability
                if 'portability_acc' in metrics[0]['post']:
                    f.write(f'portability_acc: {portability_acc}\n')


###########################################################################


def Generate_Embedding_for_IKE(args):
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        train_ds = CaptionDataset(f'data_json/{random_data_type}_train.json', config=hparams, no_image=True)
    else:
        train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams, no_image=True)
    ## Generate embedding files for IKE
    data_type = args.data_type
    sentence_model = SentenceTransformer(hparams.sentence_model_name, device=f'cuda:{hparams.device}')
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams, data_type)


####################### MiniGPT4 ##########################
def train_MEND_MiniGPT4(args):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_MEND_MiniGPT4(args):
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.run()


def train_SERAC_MiniGPT4(args):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_SERAC_MiniGPT4(args):
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_FT_MiniGPT4(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_FT_MiniGPT4_Qformer(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_qformer.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_IKE_MiniGPT4(args):
    Generate_Embedding_for_IKE(args)
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(f'data_json/{random_data_type}_eval.json', config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True
    )

    MMKE_print_result(metrics,
                      save_path=os.path.join(f'./results/{args.data_type}', 'IKE/MiniGPT4_results_portability.txt'))


####################### BLIP2 ##########################
def train_MEND_Blip2OPT(args):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_MEND_Blip2OPT(args):
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def train_SERAC_Blip2OPT(args):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_SERAC_Blip2OPT(args):
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_FT_Blip2OPT(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_FT_Blip2OPT_QFormer(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2_qformer.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_IKE_Blip2OPT(args):
    Generate_Embedding_for_IKE(args)
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(f'data_json/{random_data_type}_eval.json', config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True
    )

    MMKE_print_result(metrics,save_path=os.path.join(f'./results/{args.data_type}', 'IKE/Blip2OPT_results_portability.txt'))


####################### LLAVA ##########################
def train_MEND_LLaVA(args):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/llava.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_MEND_LLaVA(args):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/llava.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.run()


def train_SERAC_LLaVA(args):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/llava.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_SERAC_LLaVA(args):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/llava.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_FT_LLaVA(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_FT_LLaVA_mmproj(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_mmproj.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_IKE_LLaVA(args):
    Generate_Embedding_for_IKE(args)
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/llava.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    if 'random_' in args.data_type:
        random_data_type = args.data_type.replace('random_', '')
        eval_ds = CaptionDataset(f'data_json/{random_data_type}_eval.json', config=hparams, hop=args.hop)
    else:
        eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True
    )

    MMKE_print_result(metrics,
                      save_path=os.path.join(f'./results/{args.data_type}', 'IKE/LLAVA_results_portability.txt'))


####################### Qwen-VL ##########################
def train_MEND_QwenVL(args):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/qwenvl.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_MEND_QwenVL(args):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/qwenvl.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.run()


def train_SERAC_QwenVL(args):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/qwenvl.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_SERAC_QwenVL(args):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/qwenvl.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_FT_QwenVL(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwenvl.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_FT_QwenVL_ViT(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/qwenvl_vit.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_IKE_QwenVL(args):
    Generate_Embedding_for_IKE(args)
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/qwenvl.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True
    )

    MMKE_print_result(metrics,
                      save_path=os.path.join(f'./results/{args.data_type}', 'IKE/qwen-vl_results_portability.txt'))


####################### Owl-2 ##########################
def train_MEND_Owl2(args):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/owl2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_MEND_Owl2(args):
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/MEND/owl2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.run()


def train_SERAC_Owl2(args):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/owl2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    train_ds = CaptionDataset(f'data_json/{args.data_type}_train.json', config=hparams)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_SERAC_Owl2(args):
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/owl2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )

    trainer.run()


def test_FT_Owl2(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_FT_Owl2_ViT(args):
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/owl2_vit.yaml')
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.run()


def test_IKE_Owl2(args):
    Generate_Embedding_for_IKE(args)
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/owl2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    if hasattr(args, 'data_type'):
        setattr(hparams, 'data_type', args.data_type)
    eval_ds = CaptionDataset(f'data_json/{args.data_type}_eval.json', config=hparams, hop=args.hop)
    metrics, edited_model, _ = editor.edit_dataset(
        ds=eval_ds,
        train_ds='train_ds',
        keep_original_weight=True
    )

    MMKE_print_result(metrics,
                      save_path=os.path.join(f'./results/{args.data_type}', 'IKE/Owl2_results_portability.txt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--function_name', required=True, type=str, default='test_FT_Blip2OPT')
    parser.add_argument('--hop', required=True, type=int, default=1)
    parser.add_argument('--data_type', required=True, type=str, default='user')

    args = parser.parse_args()

    # Check if the function exists and is callable
    if args.function_name not in globals() or not callable(globals()[args.function_name]):
        print(f"Error: Function '{args.function_name}' does not exist.")
        sys.exit(1)

    globals()[args.function_name](args)


    #     sys.exit(1)
    # globals()[function_name]()

