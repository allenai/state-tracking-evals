import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectra import capture, replay  # noqa: E402


def random_transitions(n, heads, dim, seed=0, neg=True):
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(n, heads, dim, generator=gen, dtype=torch.float64)
    g = -torch.rand(n, heads, generator=gen, dtype=torch.float64) * 0.1
    beta = torch.rand(n, heads, generator=gen, dtype=torch.float64) * (2.0 if neg else 1.0)
    return k, g, beta


def explicit_product(k, g, beta):
    n, heads, dim = k.shape
    out = []
    for h in range(heads):
        m = torch.eye(dim, dtype=torch.float64)
        for t in range(n):
            u = k[t, h] / k[t, h].norm()
            a = math.exp(g[t, h]) * (torch.eye(dim, dtype=torch.float64) - beta[t, h] * torch.outer(u, u))
            m = a @ m
        out.append(m)
    return torch.stack(out)


def test_transition_product_matches_explicit():
    k, g, beta = random_transitions(7, 3, 5)
    got = capture.transition_product(capture._unit(k, torch.float64), g, beta)
    torch.testing.assert_close(got, explicit_product(k, g, beta))


def test_single_step_spectrum_is_alpha_and_mu():
    k, g, beta = random_transitions(1, 2, 6)
    product = capture.transition_product(capture._unit(k, torch.float64), g, beta)
    for h in range(2):
        eig = torch.linalg.eigvals(product[h]).real.sort().values
        alpha = math.exp(g[0, h])
        mu = alpha * (1 - beta[0, h].item())
        expected = torch.tensor(sorted([mu] + [alpha] * 5), dtype=torch.float64)
        torch.testing.assert_close(eig, expected)


def test_intervention_rewrites_selected_heads_only():
    g = torch.tensor([[-1e-6, -0.5], [-0.2, -1e-7]])
    beta = torch.tensor([[1.999, 0.001], [1.5, 1.995]])
    iv = capture.Intervention(snap_reflect=0.01, snap_identity=0.01, snap_alpha=1e-4, heads=[(3, 0)])
    new_g, new_beta = iv.apply(3, g, beta)
    assert new_beta[0, 0] == 2.0 and new_beta[1, 0] == 1.5  # head 0 snapped where eligible
    assert new_beta[0, 1] == pytest.approx(0.001) and new_beta[1, 1] == pytest.approx(1.995)  # head 1 untouched
    assert new_g[0, 0] == 0.0 and new_g[1, 0] == pytest.approx(-0.2)
    untouched_g, untouched_beta = iv.apply(4, g, beta)
    assert torch.equal(untouched_g, g) and torch.equal(untouched_beta, beta)
    assert capture.Intervention().is_null and capture.Intervention().name() == "none"
    power_g, _ = capture.Intervention(alpha_power=0.5).apply(0, g, beta)
    torch.testing.assert_close(power_g, g * 0.5)


def test_binned_gate_stats_counts_and_moments():
    T, H = 300, 2
    roles = np.array([replay.ROLE_CODES["target"], replay.ROLE_CODES["source"]] * (T // 2), dtype=np.int8)
    edges = capture.log2_position_edges(T)
    stats = capture.BinnedGateStats(roles, position_edges=edges, hist_position_edges=edges, n_heads=H,
                                    device=torch.device("cpu"))
    g = torch.full((T, H), -0.01)
    beta = torch.full((T, H), 2.0)
    beta[:, 1] = 0.0
    k = torch.randn(T, H, 4)
    for offset in (0, 100, 250):
        end = {0: 100, 100: 250, 250: 300}[offset]
        stats.observe(capture.Segment(0, offset, k[offset:end], g[offset:end], beta[offset:end]))
    out = stats.result()
    moments = out["moments"][0]  # (H, R, B, F)
    count = capture.GATE_FEATURES.index("count")
    neg = capture.GATE_FEATURES.index("neg")
    two = capture.GATE_FEATURES.index("beta_is_two")
    assert moments[..., count].sum() == T * H
    # Head 0 is an exact reflection everywhere, head 1 an (almost) identity.
    assert moments[0, ..., neg].sum() == T and moments[1, ..., neg].sum() == 0
    assert moments[0, ..., two].sum() == T
    assert out["hist_mu"][0].sum() == T * H
    first_bin = moments[0, replay.ROLE_CODES["target"], 0, count]
    assert first_bin == 32  # 64 tokens in [0, 64), half of them targets


def test_retention_of_exact_reflections_is_lossless():
    T, H, D = 64 * 40, 2, 8
    gen = torch.Generator().manual_seed(1)
    k = torch.randn(T, H, D, generator=gen)
    g = torch.zeros(T, H)
    beta = torch.full((T, H), 2.0)
    beta[:, 1] = 1.0  # head 1 overwrites along each key: projections, which forget
    edges = capture.log2_position_edges(T)
    ret = capture.ImpulseRetention(n_heads=H, head_dim=D, device=torch.device("cpu"), position_edges=edges,
                                   probe_every=4)
    for start in range(0, T, 1000):  # segment ends that are not block-aligned
        end = min(start + 1000, T)
        ret.observe(capture.Segment(0, start, k[start:end], g[start:end], beta[start:end]))
    out = ret.result()
    sums, counts = out["log_norm_sum"][0], out["count"][0]  # (P, H, pos, lag), (P, pos, lag)
    assert counts.sum() > 0
    mean = np.where(counts[:, None] > 0, sums / np.maximum(counts[:, None], 1), np.nan)
    assert np.nanmax(np.abs(mean[:, 0])) < 1e-9  # reflections are orthogonal
    assert np.nanmean(mean[:, 1]) < -1.0  # projections lose the write


def test_retention_matches_brute_force_product():
    T, H, D, L = 64 * 9, 1, 6, 64
    k, g, beta = random_transitions(T, H, D, seed=3)
    edges = capture.log2_position_edges(T)
    ret = capture.ImpulseRetention(n_heads=H, head_dim=D, device=torch.device("cpu"), position_edges=edges,
                                   probe_every=4)
    ret.observe(capture.Segment(0, 0, k.float(), g.float(), beta.float()))
    # The write probe started at block 4 is the largest-beta key of block 3; after all
    # 9 blocks its log-norm is log ||Phi(blocks 4..8) k||.
    block3 = slice(3 * L, 4 * L)
    t_star = 3 * L + int(beta[block3, 0].argmax())
    probe = k[t_star, 0] / k[t_star, 0].norm()
    # Lags 4 and 5 share the [4, 8) bin, so the bin holds their mean.
    lag4 = explicit_product(k[4 * L : 8 * L], g[4 * L : 8 * L], beta[4 * L : 8 * L])[0]
    lag5 = explicit_product(k[4 * L :], g[4 * L :], beta[4 * L :])[0]
    expected = 0.5 * sum(torch.log((phi @ probe).norm()).item() for phi in (lag4, lag5))
    out = ret.result()
    lag_bin = np.searchsorted(out["lag_edges_blocks"], 5, side="right") - 1
    pos_bin = capture.bin_index(np.array([4 * L]), edges)[0]
    # Probes at block 4 and block 8 both exist; only block 4's reaches lag 5.
    got = out["log_norm_sum"][0, 0, 0, pos_bin, lag_bin] / out["count"][0, 0, pos_bin, lag_bin]
    assert got == pytest.approx(expected, rel=1e-3, abs=1e-3)


def test_statement_spectra_keeps_nontrivial_eigenvalues():
    D, H = 8, 1
    u = torch.zeros(3, H, D)
    u[0, 0, 0] = 1.0
    u[1, 0, 1] = 1.0
    u[2, 0, 0] = 1.0
    g = torch.zeros(3, H)
    beta = torch.full((3, H), 2.0)
    sink = capture.StatementSpectra({5: (10, 13)}, keep=2)
    sink.observe(capture.Segment(2, 10, u, g, beta))
    out = sink.result()
    eig = np.sort_complex(out["eigenvalues"][0, 0])
    # Reflect e0, e1, e0: the product is diag(1, -1, 1, ...), so the nontrivial
    # eigenvalue is -1 (the rest equal the trivial value 1).
    assert out["trivial"][0, 0] == pytest.approx(1.0)
    assert np.isclose(eig, -1).any()


fork = pytest.importorskip("transformers.models.olmo3_5_hybrid.modeling_olmo3_5_hybrid", reason="needs the fork")


def tiny_model(layer_types):
    from transformers.models.olmo3_5_hybrid.configuration_olmo3_5_hybrid import Olmo3_5HybridConfig

    config = Olmo3_5HybridConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=len(layer_types),
        num_attention_heads=2, layer_types=layer_types, linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=16, use_peri_norm=True, pad_token_id=0, eos_token_id=1,
    )
    torch.manual_seed(0)
    return fork.Olmo3_5HybridForCausalLM(config).eval()


def test_recorder_sees_every_prompt_token_and_matches_projections():
    model = tiny_model(["linear_attention", "full_attention", "linear_attention"])
    recorder = capture.GateRecorder(model)
    seen = []
    recorder.sinks = [type("S", (), {"observe": lambda self, s: seen.append(s)})()]
    ids = torch.randint(2, 128, (1, 90))
    captured_inputs = {}
    handle = model.model.layers[0].linear_attn.register_forward_pre_hook(
        lambda m, args, kwargs: captured_inputs.update(x=kwargs["hidden_states"]), with_kwargs=True
    )
    recorder.recording = True
    with torch.inference_mode():
        model.model(input_ids=ids)
    handle.remove()
    assert sorted(s.layer for s in seen) == [0, 2]
    assert all(s.length == 90 for s in seen)
    layer0 = model.model.layers[0].linear_attn
    x = captured_inputs["x"][0].clone()
    with torch.no_grad():
        expected_beta = layer0.b_proj(x).sigmoid() * 2.0
        expected_g = -layer0.A_log.float().exp() * torch.nn.functional.softplus(layer0.a_proj(x).float() + layer0.dt_bias)
    s0 = next(s for s in seen if s.layer == 0)
    torch.testing.assert_close(s0.beta, expected_beta)
    torch.testing.assert_close(s0.g, expected_g)


def test_intervention_changes_outputs_only_when_active():
    model = tiny_model(["linear_attention", "linear_attention"])
    ids = torch.randint(2, 128, (1, 70))
    with torch.inference_mode():
        base = model(input_ids=ids).logits
        capture.GateRecorder(model, capture.Intervention())
        same = model(input_ids=ids).logits
    torch.testing.assert_close(base, same)
    model2 = tiny_model(["linear_attention", "linear_attention"])
    capture.GateRecorder(model2, capture.Intervention(snap_reflect=1.0))  # beta > 1 -> 2
    with torch.inference_mode():
        changed = model2(input_ids=ids).logits
    assert not torch.allclose(base, changed)


def test_recorder_refuses_to_wrap_twice():
    model = tiny_model(["linear_attention", "linear_attention"])
    capture.GateRecorder(model)
    with pytest.raises(RuntimeError, match="already wrapped"):
        capture.GateRecorder(model)


def test_head_ablation_zeroes_only_that_head():
    model = tiny_model(["linear_attention", "linear_attention"])
    recorder = capture.GateRecorder(model, capture.Intervention(ablate_heads=[(0, 1)]))
    outputs = {}
    module = model.model.layers[0].linear_attn
    inner = module.chunk_gated_delta_rule

    def spy(*args, **kwargs):
        out = inner(*args, **kwargs)
        outputs["o"] = out[0]
        return out

    module.chunk_gated_delta_rule = spy
    ids = torch.randint(2, 128, (1, 80))
    with torch.inference_mode():
        model(input_ids=ids)
    assert torch.count_nonzero(outputs["o"][:, :, 1]) == 0
    assert torch.count_nonzero(outputs["o"][:, :, 0]) > 0
    assert recorder.intervention.name() == "ablate_heads=0-1"


def test_attention_ablation_changes_outputs():
    base = tiny_model(["linear_attention", "full_attention"])
    ids = torch.randint(2, 128, (1, 70))
    with torch.inference_mode():
        reference = base(input_ids=ids).logits
    ablated = tiny_model(["linear_attention", "full_attention"])
    capture.GateRecorder(ablated, capture.Intervention(ablate_attention=[1])).ablate_attention(ablated)
    with torch.inference_mode():
        changed = ablated(input_ids=ids).logits
    assert not torch.allclose(reference, changed)
