# Analysis

A range of metrics are implemented here. The main entry point should be `analysis.report`, which automatically discovers
the kind of data that is present and evaluates metrics accordingly.

Several metrics require additional packages:
* R-PSNR : `kornia`
* CW-SSIM: `git+https://github.com/fbcotter/pytorch_wavelets`
* LPIPS: `lpips`
* IIP (Image Identifiability Precision scores): `lpips`
* BLEU: `datasets` (from huggingface)
* Rouge: `datasets` and `rouge-score`
* sacrebleu: `datasets` and `sacrebleu`
