# Semantic Verifier Anchor Report

Fixed expert labels are sanity checks, not a threshold calibration set.

| id | verdict | current DeBERTa fwd | current DeBERTa bwd | current DeBERTa min | STS | RoBERTa fwd | RoBERTa bwd | RoBERTa min | original | back-translated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| F1 | FAIL | 0.9922 | 0.9404 | 0.9404 | 0.8051 | 0.9884 | 0.9818 | 0.9818 | A red-haired rides a go-cart. | There's a red one in a go-kart. |
| F2 | FAIL | 0.9883 | 0.9717 | 0.9717 | 0.9634 | 0.9940 | 0.9887 | 0.9887 | The two men have just been stuck in concrete boots and are drowning in the Hudson River. | Both men are trapped in concrete boots and drowned in the Hudson River. |
| F3 | FAIL | 0.9961 | 0.9956 | 0.9956 | 0.5978 | 0.9814 | 0.9893 | 0.9814 | Nobody is performing. | No one's playing. |
| F4 | FAIL | 0.0002 | 0.0003 | 0.0002 | 0.3959 | 0.0041 | 0.0006 | 0.0006 | There is a girl dressed in light colored clothing frowning at a man. | There's a girl in brightly colored clothes who's rubbing a man. |
| F5 | FAIL | 0.1976 | 0.8638 | 0.1976 | 0.8459 | 0.9682 | 0.9831 | 0.9682 | A worker finishes work and leaves the office. | A worker retires from work. |
| F6 | FAIL | 0.0005 | 0.0011 | 0.0005 | 0.2183 | 0.6746 | 0.9500 | 0.6746 | Two people are hugging. | Two people are kissing. |
| F7 | FAIL | 0.9839 | 0.8687 | 0.8687 | 0.9120 | 0.8293 | 0.8930 | 0.8293 | A male is leaping above white water. | A male jumps on white water. |
| F8 | FAIL | 0.8193 | 0.8770 | 0.8193 | 0.9914 | 0.9785 | 0.9037 | 0.9037 | Two young boys wearing collared shirts and baseball caps have their blue-tinted tongues stuck out. | Two young boys wearing collared shirts and baseball caps have deep blue tongues. |
| P1 | PASS | 0.9971 | 0.9971 | 0.9971 | 0.9966 | 0.9931 | 0.9932 | 0.9931 | Nobody has food. | No one has food. |
| P2 | PASS | 0.9980 | 0.9951 | 0.9951 | 0.9786 | 0.9940 | 0.9926 | 0.9926 | The girl is not wearing shoes. | The girl doesn't wear shoes. |
| P3 | PASS | 0.9932 | 0.9966 | 0.9932 | 0.9967 | 0.9932 | 0.9933 | 0.9932 | Two girls are playing outside. | Both girls are playing outside. |
| P4 | PASS | 0.9966 | 0.9966 | 0.9966 | 0.9964 | 0.9936 | 0.9938 | 0.9936 | The box contains 5 apples. | The box contains five apples. |
| P5 | PASS | 0.9980 | 0.9971 | 0.9971 | 0.9962 | 0.9933 | 0.9925 | 0.9925 | One lone skier crosses the slope. | A single skier crosses the slope. |

## Models and runtimes

- Current DeBERTa: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, revision 6f5cf0a2b59cabb106aca4c287eed12e357e90eb, anchor runtime 3.25s.
- STS: cross-encoder/stsb-roberta-base, revision d576534b67143e2c70ee9966d7fdbf5835728d13, anchor runtime 0.22s.
- RoBERTa MNLI: FacebookAI/roberta-large-mnli, revision 2a8f12d27941090092df78e4ba6f0928eb5eac98, anchor runtime 0.78s.
- BART MNLI: not run (optional model; no result is inferred).
