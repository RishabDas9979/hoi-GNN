# Viva preparation: likely questions, honest answers

Fill the numbers from `results/` only.

1. **Why a graph?** A person's action depends on context (other people, nearby objects, relative position). Message passing lets a pair's score use that context; the MLP baseline sees each pair alone. Whether it helps is an empirical question answered by the comparison table.
2. **Why multi-label with sigmoid?** One person can hold and look at the same object. Softmax forces a single winner.
3. **Why GATv2 over GAT?** GATv2 computes dynamic attention (the ranking of neighbours can depend on the query node); GAT's static attention cannot. Edge attributes enter the attention score.
4. **What are the 9 edge features?** dx, dy, distance, log width ratio, log height ratio, log area ratio, IoU, human area fraction, object area fraction.
5. **How do you handle actions with no object?** Agent-only actions (stand, walk, run, smile) are masked on human-object pairs. Positives with no annotated object are masked, not treated as negatives.
6. **Ground-truth boxes vs detector boxes?** GT boxes isolate the interaction model; detector boxes include detection errors. They are reported separately and not compared.
7. **Is your mAP the official V-COCO metric?** It is our reimplementation; say whether it was diffed against `s-gupta/v-coco`. Do not call it official unless it was.
8. **Did the graph model beat the MLP?** Answer from the table, including if it did not, and give the seed spread.
9. **How reliable are the explanations?** Fidelity+/-, sparsity, stability and the randomised-model check measure faithfulness to the model, not to reality.
10. **Does GNNExplainer show which pixels or body parts mattered?** No. It ranks entities and spatial edges. Attention weights are not proof of importance.
11. **What are the limits of your evidence?** Subset size, number of seeds, one dataset, own AP code, small explanation sample.
12. **Did any run use synthetic data?** Yes for pipeline tests; they are never reported as results.
13. **Did training see real image features?** Answer only after checking images were found (the zero-feature fallback in the dataset class).
14. **What would you do next?** HICO-DET, Faster R-CNN backend, multi-seed ablations, official evaluator diff, pixel-level explanations (separate method).
