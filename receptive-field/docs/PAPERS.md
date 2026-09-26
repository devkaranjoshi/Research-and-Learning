# PAPERS — dial-to-paper reference (authoritative)

Every experiment must trace to a primary paper here. Cite the specific claim in the *Hypothesis* field of the runlog. New dials add their paper here **first**.

## Part A — Convolutional receptive field

| Key | Paper | Claim we use | Nodes |
|-----|-------|--------------|-------|
| Luo2016 | Luo, Li, Urtasun, Zemel — *Understanding the Effective Receptive Field in Deep CNNs* (NeurIPS 2016) | ERF grows ∝ √depth, central Gaussian fraction of TRF; expands during training | N2, N4, N5, N10 |
| Araujo2019 | Araujo, Norouzi, Vasconcelos — *Computing Receptive Fields of CNNs* (Distill 2019) | closed-form TRF recurrence incl. stride/dilation/pooling | N0, N1, N6, N7, N9 |
| YuKoltun2016 | Yu & Koltun — *Multi-Scale Context Aggregation by Dilated Convolutions* (ICLR 2016) | dilation expands RF exponentially without resolution loss; gridding artifacts | N8 |
| He2016 | He et al. — *Deep Residual Learning* (CVPR 2016) | skip connections shrink ERF relative to TRF (path-skipping) | N12 |
| Ronneberger2015 | Ronneberger et al. — *U-Net* (MICCAI 2015) | encoder-decoder RF for dense prediction | N12 |

## Part B — Attention receptive field

| Key | Paper | Claim we use | Nodes |
|-----|-------|--------------|-------|
| Vaswani2017 | Vaswani et al. — *Attention Is All You Need* (NeurIPS 2017) | full self-attention = global TRF; causal mask = backward RF | T0, T2 |
| Clark2019 | Clark et al. — *What Does BERT Look At?* (BlackboxNLP 2019) | heads specialize (local, delimiter, syntactic) | T1, T5 |
| Kobayashi2020 | Kobayashi et al. — *Attention Is Not Only a Weight* (EMNLP 2020) | norm-weighting changes the apparent RF vs raw weights | T1 (ruler B) |
| AbnarZuidema2020 | Abnar & Zuidema — *Quantifying Attention Flow in Transformers* (ACL 2020) | rollout/flow accounts for residual stream across layers | T1 (ruler C) |
| Elhage2021 | Elhage et al. — *A Mathematical Framework for Transformer Circuits* (Anthropic 2021) | previous-token heads; patching as causal RF measure | T0, T5 (ruler D) |
| Olsson2022 | Olsson et al. — *In-context Learning and Induction Heads* (Anthropic 2022) | induction heads emerge in a training phase change; RF = last occurrence | T5, T12 |
| VigBelinkov2019 | Vig & Belinkov — *Analyzing the Structure of Attention* (BlackboxNLP 2019) | RF/attention structure varies systematically with layer depth | T6 |
| Michel2019 | Michel, Levy, Neubig — *Are Sixteen Heads Really Better Than One?* (NeurIPS 2019) | head redundancy / specialization; many prunable | T7 |
| Su2021 | Su et al. — *RoFormer: Rotary Position Embedding* (2021) | RoPE injects relative-distance structure (locality bias) | T8 |
| Press2021 | Press, Smith, Lewis — *ALiBi: Train Short, Test Long* (ICLR 2022) | linear distance penalty → strong locality prior, length extrapolation | T8 |
| Beltagy2020 | Beltagy, Peters, Cohan — *Longformer* (2020) | window + dilation + global tokens; stacked RF ≈ w×L | T9, T10 |
| Zaheer2020 | Zaheer et al. — *Big Bird* (NeurIPS 2020) | window + random + global recovers full-attention expressivity | T10 |
| Xiao2023 | Xiao et al. — *Efficient Streaming LMs with Attention Sinks (StreamingLLM)* (ICLR 2024) | massive mass on initial tokens; window severs sink → collapse; re-inject fixes | T4, T11 |
| Liu2023 | Liu et al. — *Lost in the Middle* (TACL 2024) | effective context ≪ architectural context; U-shaped position recall | T13 |
| Mistral2023 | Jiang et al. — *Mistral 7B* (2023) | causal SWA window=4096 over 32 layers ⇒ reach ~L×W=131k; rolling-buffer KV cache; GQA | Task1, Task2 |

## Methodological note — two approximation philosophies (T9 thesis)
- **Approximate the matrix:** Linformer (Wang 2020, low-rank), Performer (Choromanski 2021, kernel) — reconstruct the attention output numerically.
- **Approximate the ERF:** SWA family (Beltagy2020, Xiao2023, Zaheer2020) — bet that effective RF is local + depth-composable. *This sandbox evaluates the second via ERF, not weight-reconstruction error.*
