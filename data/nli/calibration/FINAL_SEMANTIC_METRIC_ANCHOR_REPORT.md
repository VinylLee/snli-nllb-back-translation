# Final Semantic Metric Anchor Report

These are targeted expert anchors, not a benchmark and not a threshold-calibration set.

| id | verdict | DeBERTa min | STS-base | STS-large | BLEURT raw | RoBERTa min | original | candidate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| F1 | FAIL | 0.9404296875 | 0.8051077723503113 | 0.7982110977172852 | -0.7591378092765808 | 0.9818136692047119 | A red-haired rides a go-cart. | There's a red one in a go-kart. |
| F2 | FAIL | 0.9716796875 | 0.9633954763412476 | 0.9269264936447144 | 0.6399165987968445 | 0.9887322187423706 | The two men have just been stuck in concrete boots and are drowning in the Hudson River. | Both men are trapped in concrete boots and drowned in the Hudson River. |
| F3 | FAIL | 0.99560546875 | 0.5978107452392578 | 0.513242781162262 | 0.3049829602241516 | 0.9814233183860779 | Nobody is performing. | No one's playing. |
| F4 | FAIL | 0.0001817941665649414 | 0.3958658277988434 | 0.42916637659072876 | -0.5698796510696411 | 0.0005981291178613901 | There is a girl dressed in light colored clothing frowning at a man. | There's a girl in brightly colored clothes who's rubbing a man. |
| F5 | FAIL | 0.1976318359375 | 0.8459407091140747 | 0.7425205707550049 | 0.4127765893936157 | 0.9682340025901794 | A worker finishes work and leaves the office. | A worker retires from work. |
| F6 | FAIL | 0.0004794597625732422 | 0.21827539801597595 | 0.3802531957626343 | 0.33020901679992676 | 0.6745941638946533 | Two people are hugging. | Two people are kissing. |
| F7 | FAIL | 0.86865234375 | 0.9119795560836792 | 0.8959492444992065 | 0.5228309631347656 | 0.8292924165725708 | A male is leaping above white water. | A male jumps on white water. |
| F8 | FAIL | 0.8193359375 | 0.9914376139640808 | 0.9432818293571472 | 0.5330696105957031 | 0.9036691784858704 | Two young boys wearing collared shirts and baseball caps have their blue-tinted tongues stuck out. | Two young boys wearing collared shirts and baseball caps have deep blue tongues. |
| P1 | PASS | 0.9970703125 | 0.9966153502464294 | 0.9633382558822632 | 1.0286519527435303 | 0.9930799603462219 | Nobody has food. | No one has food. |
| P2 | PASS | 0.9951171875 | 0.9786160588264465 | 0.9669901728630066 | 0.7560535669326782 | 0.9925745725631714 | The girl is not wearing shoes. | The girl doesn't wear shoes. |
| P3 | PASS | 0.9931640625 | 0.9967299699783325 | 0.9654228091239929 | 0.591213047504425 | 0.993241548538208 | Two girls are playing outside. | Both girls are playing outside. |
| P4 | PASS | 0.99658203125 | 0.9964082837104797 | 0.9680079817771912 | 1.0403209924697876 | 0.9936327338218689 | The box contains 5 apples. | The box contains five apples. |
| P5 | PASS | 0.9970703125 | 0.9962437152862549 | 0.9691234827041626 | 0.8036518692970276 | 0.9925300478935242 | One lone skier crosses the slope. | A single skier crosses the slope. |

## Models and runtimes

- Current DeBERTa: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, revision 6f5cf0a2b59cabb106aca4c287eed12e357e90eb; existing/current anchor scores.
- STS-base: cross-encoder/stsb-roberta-base; existing scores reused.
- STS-large: cross-encoder/stsb-roberta-large, revision 2b12c2c0088918e76151fd5937b7bba986ef1f98; runtime 0.37s.
- BLEURT: Elron/bleurt-base-512, revision 4f4abeeba7c29ded45fc90b8a66eb49c8569f587; raw-score runtime 0.12s.
- BLEURT primary fallback note: ValueError: The checkpoint you are trying to load has model type `bleurt` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date. You can upda.
- RoBERTa-MNLI: existing scores reused; BART-MNLI was not run.
