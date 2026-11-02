# Deployment note

- **Ship inference only.** Deploy `app.py` with a small checkpoint (`best.pt` from one run, a few MB) and
  `yolov8s.pt`. Do not train on a free host.
- **Needed files:** `app.py`, `config.yaml`, `configs/`, `data/`, `detection/`, `models/`, `explainability/`,
  `visualization/`, `utils/`, `evaluation/aggregate.py`, `evaluation/ablation.py`, `requirements.txt`,
  and the `results/` folder holding the checkpoint, `metrics.json` and `history.json` of the runs to show.
- **Hosting:** Streamlit Community Cloud is CPU-only with limited memory. Expect a few seconds per image
  (detector on CPU) and slow GNNExplainer; lower "optimisation steps" in the app. Use a CPU torch wheel
  in the hosted `requirements.txt`, not the `+cu121` build.
- **Local demo is the safe plan** for the viva: `streamlit run app.py` on the laptop with the checkpoint in `results/`.
- **Before demoing:** confirm the selected checkpoint is labelled V-COCO, and that images were read from disk
  during training (see the dummy-feature warning in the README).
- Pin versions with `pip freeze > requirements-lock.txt` after a successful run.
