# Run this after training has completed

shopt -s globstar
./set_kaldi_root.sh
for x in */exp/*/*/decode*; do [ -d $x ] && grep WER $x/wer_* | pytorch-kaldi-fl/utils/best_wer.sh ; done
