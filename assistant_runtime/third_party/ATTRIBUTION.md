# Third-party citation checker

The optional support service uses MiniCheck by Liyan Tang, Philippe Laban and Greg Durrett, described in *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents*, EMNLP 2024. It does not claim that SISU developed or trained this pretrained model.

- Upstream code: https://github.com/Liyan06/MiniCheck at commit `b58b9fa69acbd1015ec970fa65dd752413a053d2`.
- Model: https://huggingface.co/lytang/MiniCheck-Flan-T5-Large at revision `96eafd01cee2d16cf81aaa2fb226b14f422a37b3`.
- Paper: https://aclanthology.org/2024.emnlp-main.499/.

The pinned upstream Apache 2.0 code license is retained in `MiniCheck_CODE_LICENSE.txt`. The pinned model card, which declares MIT licensing, is retained in `MiniCheck_Flan_MODEL_CARD.md`. The model files stay in the recorded workspace cache, with exact byte hashes in their download manifest.

SISU's helper scripts adapt the loading environment for offline CPU execution, verified artifact hashes, fixed package versions and bounded requests. Only the reviewed upstream scoring methods are loaded; the upstream automatic device-selection constructor is not used. These adaptations and the surrounding claim planner, process isolation, service and user interface are part of the SISU artifact. They do not reproduce the authors' benchmark results.
