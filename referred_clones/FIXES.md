# referred_clones — provenance & compatibility fixes

Each repo was cloned `--depth 1`, its `.git/` stripped, and heavy artifacts are git-ignored.
Each runnable baseline needs its **own isolated venv/conda env** — do NOT mix with the main `myenv`.

## VLP-MABSA
- repo: https://github.com/NUSTM/VLP-MABSA
- cloned commit: `04fba6c4e6f537c519c174c0c605ce5059f092a8`
- original stack: torch1.6, transformers3.4, BART, Faster-RCNN
- fixes needed:
  - [ ] migrate transformers 3.4->4.4x (TokenizerFast attrs)
  - [ ] replace fastnlp
  - [ ] pin h5py>=3
  - [ ] canonical data/feature source: download feats from its Drive/Baidu (code d0tn)

## JML
- repo: https://github.com/MANLP-suda/JML
- cloned commit: `ad277d9087340a66d1bade1c5b0c4d276fd4cbbd`
- original stack: py3.6, torch1.1, BERT+Mask/Faster-RCNN
- fixes needed:
  - [ ] torch 1.1->2.x rewrite
  - [ ] transformers API migration
  - [ ] pin numpy<2
  - [ ] 17GB supplementary on Baidu (code 53ej); RCNN feats may need detectron2 backport

## AoM
- repo: https://github.com/SilyRab/AoM
- cloned commit: `82b5ed0a9cdfd602745953b401d5122a8437103b`
- original stack: builds on VLP-MABSA, transformers3.4
- fixes needed:
  - [ ] same VLP-MABSA dep chain
  - [ ] reuses VLP-MABSA 36-region 2048-d feats
  - [ ] expect deprecated-API rewrites

## M2DF
- repo: https://github.com/grandchicken/M2DF
- cloned commit: `51ee0390f84c1575136fb586026b8f97a36dc28d`
- original stack: py3.7.13, torch1.12, transformers3.4
- fixes needed:
  - [ ] torch1.12 OK on T4
  - [ ] pin numpy 1.24, fastnlp 0.6/0.7
  - [ ] h5py wheels
  - [ ] download feats from Drive

## CMMT
- repo: https://github.com/yangli-hub/CMMT-Code
- cloned commit: `06d38658b0c0e3c585ca8fa17d354c353c676e84`
- original stack: py3.7, torch1.0, RoBERTa+ResNet152+CRF
- fixes needed:
  - [ ] torch 1.0->1.13+
  - [ ] replace pytorch-crf 0.7.2 -> torchcrf
  - [ ] transformers 3.4 pin
  - [ ] CoNLL data + ResNet-152 weights

## MultiPoint
- repo: https://github.com/YangXiaocui1215/MultiPoint
- cloned commit: `6f2f22ef15f1faf1229f8d177af09e3359113c13`
- original stack: py3.8+, torch1.9+, roberta-large+NF-ResNet50
- fixes needed:
  - [ ] most modern of the set
  - [ ] align sentence-transformers/timm to torch
  - [ ] Drive data

## DQPSA
- repo: https://github.com/pengts/DQPSA
- cloned commit: `8bbc9d5c05ea96341c99e3b9a571b3e18e89c5f3`
- original stack: torch1.13, accelerate+deepspeed, spaCy3.5
- fixes needed:
  - [ ] pin transformers<=4.26
  - [ ] upgrade accelerate/deepspeed or keep torch1.13
  - [ ] data+ckpts on Baidu (code 2024)
  - [ ] CPU: strip deepspeed

## TCMT
- repo: https://github.com/ZouWang-spider/TCMT
- cloned commit: `4dd3164ae5fff7d24e590233b83abef94c3f3b3c`
- original stack: torch~1.13; YOLOv5+ViT-GPT2+Tesseract+FITE
- fixes needed:
  - [ ] install Tesseract (system) + pytesseract
  - [ ] YOLOv5 pin torch<=1.13
  - [ ] FITE not public (may block full repro)

## VLHA
- repo: https://github.com/ZouWang-spider/VLHA
- cloned commit: `39c39a7b3e68b59fafd2088422c592d30ba37d3e`
- original stack: Scene-Graph-Benchmark.pytorch + BiAffine
- fixes needed:
  - [ ] SGB.pytorch unmaintained -> pin torch<=1.13
  - [ ] BiAffine needs Cython build
  - [ ] requirements.txt 404: reverse-engineer deps

## TomBERT
- repo: https://github.com/jefferyYu/TomBERT
- cloned commit: `31bc79fb9a913a2480d5a56ffe5009d986d6eb2a`
- original stack: py3.7, torch1.0, BERT+ResNet-152
- fixes needed:
  - [ ] torch 1.0->2.x rewrite
  - [ ] MASC data source (absa_data, 49 region feats)
  - [ ] transformers API migration

## UMT
- repo: https://github.com/jefferyYu/UMT
- cloned commit: `f478ee65564b5532319a6cc643203149acfc5eed`
- original stack: py3.7, torch1.0, BERT+ResNet-152+CRF
- fixes needed:
  - [ ] as TomBERT + pytorch-crf pin
  - [ ] MNER collapsed baselines

## Atlantis
- repo: https://github.com/Xillv/Atlantis
- cloned commit: `2a27d9ced7123f4feb5384d2b23229730c3db3ef`
- original stack: py3.9, torch1.12.1, transformers4.32, FLAN-T5
- fixes needed:
  - [ ] nearly modern: bump transformers->4.35, CUDA 11.3->11.8
  - [ ] see sibling Chimera repo for full env

## CORSA
- repo: https://github.com/Liuxj-Anya/CORSA
- cloned commit: `169b892e5bc63e349e191529dff8337cf45fe999`
- original stack: py3.8+, torch1.9+, YOLO/Faster-RCNN
- fixes needed:
  - [ ] builds C-MABSA from Twitter
  - [ ] torchvision 0.15+ for RCNN
  - [ ] CPU bbox dynamic shapes

## MCPL
- repo: https://github.com/qujiaqi-babu/MCPL
- cloned commit: `896cbdfe45ce111b4a05df66b150df0b723878c0`
- original stack: BERT; minimal README
- fixes needed:
  - [ ] uses CopotronicRifat/TwitterDataMABSA
  - [ ] check requirements.txt in clone
  - [ ] bottom-up RCNN feats

## RpBERT
- repo: https://github.com/Multimodal-NER/RpBERT
- cloned commit: `14318f5f8d744d4dda723cf2dcb0b1132ccb8764`
- original stack: BERT+ResNet-101+BiLSTM-CRF
- fixes needed:
  - [ ] torch>=1.9, transformers>=4
  - [ ] ResNet-101 via torchvision
  - [ ] MNER baseline

## Cite-only (no public code found — use paper numbers)
- SGBIS (KBS 2025/26), RNG (ICME 2024), Vanessa (EMNLP-F 2024), DSEM (ICMR 2025)
## Text-only (SemEval, cited)
- SPAN (huminghao16/SpanABSA), D-GCN (cuhksz-nlp/DGSA)
