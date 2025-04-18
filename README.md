

<h1 align="center"> <a href="https://arxiv.org/pdf/2502.19870">【ICLR 2025 🔥】MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge</a></h1>
<h5 align="center">


[![arXiv](https://img.shields.io/badge/Arxiv-2502.19870-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2502.19870)  [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MMKE_Bench-blue)](https://huggingface.co/datasets/kailinjiang/MMKE-Bench-dataset)  [![paperwithcode](https://img.shields.io/badge/PWC-MMKE_Bench-blue?logo=paperswithcode)](https://paperswithcode.com/paper/mmke-bench-a-multimodal-editing-benchmark-for) [![code](https://img.shields.io/badge/Code-MMKE_Bench-blue?logo=github)](https://github.com/MMKE-Bench-ICLR/MMKE-Bench) [![website](https://img.shields.io/badge/Website-MMKE_Bench-orange?logo=homepage)](https://mmke-bench-iclr.github.io/) [![Slides PDF](https://img.shields.io/badge/Slides-PDF-ff1493?logo=slideshare)](MMKE-Bench.pdf) [![airchina](https://img.shields.io/badge/数源AI-MMKE_Bench-red?logo=airchina)](https://mp.weixin.qq.com/s/iN826lITi5Xyz-3GnrdVIQ) [![codeproject](https://img.shields.io/badge/量子之心-MMKE_Bench-red?logo=codeproject)](https://www.xiaohongshu.com/explore/67e2d622000000000603cbfc?note_flow_source=wechat&xsec_token=CBldN8wUavDAzFvP4tK_noXO94RAXcelKKqlO3pFiJ6EQ=) [![actix](https://img.shields.io/badge/极市平台-MMKE_Bench-red?logo=actix)](https://mp.weixin.qq.com/s/JfxeytzWU0QoIUfJTGqgQQ) [![zhihu](https://img.shields.io/badge/知乎-MMKE_Bench-red?logo=zhihu)](https://zhuanlan.zhihu.com/p/30599722521)


</h5>



## Table of Contents

- [Table of Contents](#table-of-contents)
- [🔔 News](#-news)
- [🌟 Overview](#overview)
- [🤗 Dataset](#-dataset)
- [🛠️ Requirements and Installation](#️-requirements-and-installation)
- [💥 Training](#training)
- [✏️ Citation](#️-citation)
- [⭐ Star History](#star-history)
- [🎉Contributors](#contributors)





## 🔔 News
* **[2024.10.30]**  The **Project Page** is already on display！ [Project Page](https://mmke-bench-bigai.github.io/).

* **[2024.10.25]**  **Code** is available now!

* **[2024.10.25]**  We release the **MMKE-Bench dataset** at 🤗 [Huggingface Dataset](https://huggingface.co/datasets/kailinjiang/MMKE-Bench-dataset).

* **[2023.10.25]**  We hung the **paper** on 🤗 [Huggingface Papers](https://huggingface.co/papers/2406.11194).

  

## 🌟Overview

**TL;DR:** We propose <span style="color:brown">**MMKE-Bench**</span>, a challenging benchmark for evaluating diverse semantic editing in real-world scenarios.

<img src="figs\fig1.jpg" width="900px">

<p align="justify">
    <strong><span style="color:black">Overview</span> of the <span style="color:brown">MMKE-Bench</span> dataset. Our contribution can be summarized as follows:</strong>
</p>


<p align="justify" style="margin-left: 30px; text-indent: -30px;">
    <strong><span style="color:black">1) Overview of MMKE-Bench</span></strong>: 
    MMKE-Bench is introduced as a benchmark designed to test semantic editing capabilities in realistic scenarios. It utilizes natural language for knowledge representation and includes three editing types aligned with practical contexts.
</p>


<p align="justify" style="margin-left: 30px; text-indent: -30px;">
    <strong><span style="color:black">2) Development of the Benchmark Pipeline</span></strong>: 
    Describes the novel pipeline used to develop the benchmark, which includes collecting original knowledge, generating editable knowledge, and crafting evaluation questions based on specific principles.
</p>


<p align="justify" style="margin-left: 30px; text-indent: -30px;">
    <strong><span style="color:black">3) Experimental Analysis and Challenges</span></strong>: 
    Details extensive experiments with various standard methods and large language models, highlighting several limitations in the existing approaches to knowledge editing in both single and multiple edit scenarios.
</p>



## 🤗 Dataset

<p align="justify">
We introduce <strong><span style="color:brown">MMKE-Bench</span></strong>, a benchmark designed to evaluate the ability of LMMs to edit visual knowledge in real-world scenarios. <strong><span style="color:brown">MMKE-Bench</span></strong> incorporates three editing tasks: <strong><span style="color:brown">visual entity editing</span></strong>, <strong><span style="color:brown">visual semantic editing</span></strong>, and <strong><span style="color:brown">user-specific editing</span></strong>. Additionally, it uses free-form natural language to represent and edit knowledge, offering more flexibility. The benchmark includes <strong><span style="color:brown">2,940</span></strong> pieces of knowledge and <strong><span style="color:brown">7,229</span></strong> images across 110 fine-grained types, with automatically generated, human-verified evaluation questions.
</p>


You can download **MMKE-Bench data** 🤗 [Huggingface Dataset](https://huggingface.co/datasets/kailinjiang/MMKE-Bench-dataset). And the expected structure of files is:

```text
MMKE-Bench
|-- data_json
|   |-- entity
|   |   |-- train.json
|   |   |-- eval.json
|   |-- visual
|   |   |-- train.json
|   |   |-- eval.json
|   |-- user
|   |   |-- train.json
|   |   |-- eval.json
|-- data_image
|   |-- entity
|   |   |-- image.....
|   |-- visual
|   |   |-- image.....
|   |-- user
|   |   |-- image.....
```



## 🛠️ Requirements and Installation

```text
# clone MMKE-Bench
git clone https://github.com/mmke-bench-bigai/mmke-bench.git

cd mmke-bench

# create conda env
#for blip2 llava minigpt4 in FT IKE MEND SERAC 
conda env create -f envs/mmke.yml

#for blip2 llava minigpt4 in KE
conda env create -f envs/mmke-ke.yml

```



## 💥Training

**For FT-LLM:**

```shell
python multimodal_edit.py --function_name=test_FT_LLaVA --hop=1 --data_type=entity

function_name in ['test_FT_LLaVA','test_FT_MiniGPT4','test_FT_Blip2OPT']
data_type in ['entity','visual','user']
```

**For FT-Alignment:**

```shell
python multimodal_edit.py --function_name=test_FT_LLaVA_mmproj --hop=1 --data_type=entity

function_name in ['test_FT_LLaVA_mmproj','test_FT_MiniGPT4_Qformer','test_FT_Blip2OPT_QFormer']
data_type in ['entity','visual','user']
```

**For SERAC:**

```shell
python multimodal_edit.py --function_name=train_SERAC_LLaVA --hop=1 --data_type=entity

function_name in ['train_SERAC_LLaVA','train_SERAC_MiniGPT4','train_SERAC_Blip2OPT']
data_type in ['entity','visual','user']
```

**For MEND:**

```shell
python multimodal_edit.py --function_name=train_MEND_LLaVA --hop=1 --data_type=entity

function_name in ['train_MEND_LLaVA','train_MEND_MiniGPT4','train_MEND_Blip2OPT']
data_type in ['entity','visual','user']
```

**For IKE:**

```shell
python multimodal_edit.py --function_name=test_IKE_LLaVA --hop=1 --data_type=entity

function_name in ['test_IKE_LLaVA','test_IKE_MiniGPT4','test_IKE_Blip2OPT']
data_type in ['entity','visual','user']
```

**For KE:**

```shell
bash KE/train_ke.sh 0 llava entity
bash KE/train_ke.sh 0 minigpt4 entity
bash KE/train_ke.sh 0 blip2 entity

model_name in ['llava','minigpt4','blip2']
data_type in ['entity','visual','user']

bash KE/test_multihop.sh 0 llava 1 entity
bash KE/test_multihop.sh 0 minigpt4 1 entity
bash KE/test_multihop.sh 0 blip2 1 entity
```

Editing GPU memory usage 【memory/res_max_val】
|     Entity   |  BLIP2-OPT | LLaVA-1.5| MiniGPT-4|
|:------------:|:----------:|:--------:|:--------:|
|    FT-LLM    |    21GB    |   35GB   |   45GB   | 
| FT-Alignment |    24GB    |   40GB   |   50GB   | 
|     SERAC    |    17GB    |   80GB   |   67GB   | 
|      IKE     |    14GB    |   50GB   |   26GB   | 
|     MEND     |    23GB    |   61GB   |   43GB   | 

|     Visual   |  BLIP2-OPT | LLaVA-1.5| MiniGPT-4|
|:------------:|:----------:|:--------:|:--------:|
|    FT-LLM    |    21GB    |   35GB   |   45GB   | 
| FT-Alignment |    23GB    |   39GB   |   48GB   | 
|     SERAC    |    16GB    |   73GB   |   58GB   |  
|      IKE     |    15GB    |   25GB   |   25GB   | 
|     MEND     |    21GB    |   55GB   |   40GB   |

|     User     |  BLIP2-OPT | LLaVA-1.5| MiniGPT-4|
|:------------:|:----------:|:--------:|:--------:|
|    FT-LLM    |    21GB    |   35GB   |   45GB   |  
| FT-Alignment |    23GB    |   38GB   |   48GB   | 
|     SERAC    |    15GB    |   71GB   |   56GB   | 
|      IKE     |    23GB    |   30GB   |   28GB   | 
|     MEND     |    21GB    |   54GB   |   39GB   |


Here we put under 'hugging_cache' folder and 'openai' folder:
```bash
# models in hugging_cache folder
hugging_cache/
├── all-MiniLM-L6-v2/
├── bert-base-uncased/
├── distilbert-base-cased/
├── Llama-2-7b-hf/
├── llava-v1.5-7b/
├── mplug-owl2-llama2-7b/
├── opt-2.7b/
├── opt-125m/
├── vicuna-7b/
├── vicuna-7b-v1.5/
│   
├── blip2_pretrained_flant5xxl.pth
├── blip2_pretrained_opt2.7b.pth
├── eva_vit_g.pth
└── pretrained_minigpt4_7b.pth

# clip-vit model in openai folder
openai/
└── clip-vit-large-patch14-336/
``` 
Links are in the following:
<table>
    <tr>
        <td><a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">all-MiniLM-L6-v2</a></td>
        <td><a href="https://huggingface.co/google-bert/bert-base-uncased">bert-base-uncased</a></td>
        <td><a href="https://huggingface.co/distilbert/distilbert-base-cased">distilbert-base-cased</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/liuhaotian/llava-v1.5-7b">llava-v1.5-7b</a></td>
        <td><a href="https://huggingface.co/facebook/opt-2.7b">opt-2.7b</a></td>
        <td><a href="https://huggingface.co/facebook/opt-125m">opt-125m</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main">vicuna-7b</a></td>
        <td><a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">vicuna-7b-v1.5</a></td>
        <td><a href="https://huggingface.co/NousResearch/Llama-2-7b-hf">Llama-2-7b-hf</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b">mplug-owl2-llama2-7b</a></td>
        <td><a href="https://huggingface.co/spaces/Vision-CAIR/minigpt4/blob/main/blip2_pretrained_flant5xxl.pth">blip2_pretrained_flant5xxl.pth</a></td>
        <td><a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_opt2.7b.pth">blip2_pretrained_opt2.7b.pth</a></td>
    </tr>
    <tr>
        <td><a href="https://huggingface.co/spaces/Vision-CAIR/minigpt4/blob/main/prerained_minigpt4_7b.pth">prerained_minigpt4_7b.pth</a></td>
        <td><a href="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth">eva_vit_g.pth</a></td>
        <td><a href="https://huggingface.co/openai/clip-vit-large-patch14-336">clip-vit-large-patch14-336</a></td>
    </tr>
</table>

<p align="right">(<a href="#readme-top">back to top</a>)</p>




## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```text
@article{du2024mmke_bench,
            title = {MMKE-Bench: A Multimodal Editing Benchmark for Diverse Visual Knowledge},
            author = {Yuntao Du and Kailin Jiang and Zhi Gao and Chenrui Shi and Zilong Zheng and Siyuan Qi and Qing Li},
            year = {2024}
          }
```



## ⭐ Star History


[![Star History Chart](https://api.star-history.com/svg?repos=MMKE-Bench-ICLR/MMKE-Bench&type=Date)](https://star-history.com/#MMKE-Bench-ICLR/MMKE-Bench&Date)




## 🎉Contributors


<a href="https://github.com/MMKE-Bench-ICLR/MMKE-Bench/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=MMKE-Bench-ICLR/MMKE-Bench" />
</a>




