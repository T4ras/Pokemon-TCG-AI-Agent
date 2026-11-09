> Public review copy of a retained historical report. Account names, local paths, artifact fingerprints and operational commands have been sanitized. Scientific results, failures and limitations are retained; this is not a new evaluation.

# RL G round30: OceanMix argmax leaderboard submission

Submitted UTC: [removed]. Submission ID: 57239424.

Model: G round30 from stage-three confirmation. Same OceanMix deck. CPU FP32 conditional greedy argmax at each selection step, with legal masks and STOP. Model weights are identical to the source checkpoint; optimizer/RNG state is retained locally and excluded from the inference archive. No retraining.

Validation: 80 tests; four native BO3; 607 decisions identical to the training implementation using greedy decoding; two official-environment BO3 through packaged main.py. No native failures. These checks validate execution, not a win-rate estimate.

Archive SHA256: `[artifact fingerprint omitted]`. Size: 11549258 bytes.

Kaggle record: `{"ref": 57239424, "totalBytes": 11549258, "date": "[removed]", "description": "RL G round30 | OceanMix | GREEDY ARGMAX | stage3", "fileName": "submission.tar.gz", "submittedBy": "portfolio-owner", "submittedByRef": "portfolio-owner", "teamName": "Taras Harbar", "url": "/submissions/57239424/57239424.raw"}`.

[Competition submissions](https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground/submissions)

The previous60.2% BO3 confirmation used sampled actions. This greedy agent has a different inference policy; its leaderboard strength is measured separately.

## Latest leaderboard status

Checked UTC: [removed]. Kaggle status: COMPLETE. Public score: **425.7**. Submission ID: 57239424. Leaderboard ratings can change as further matches are played.
