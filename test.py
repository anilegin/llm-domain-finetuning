# test_bertscore.py — quick smoke test for the torch.load CVE check
print("Testing BERTScorer initialization...")
try:
    from bert_score import BERTScorer
    scorer = BERTScorer(
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        rescale_with_baseline=True,
    )
    P, R, F1 = scorer.score(["hello world"], ["hello world"])
    print(f"✅ BERTScorer works! F1={F1.item():.4f}")
except ValueError as e:
    if "torch.load" in str(e) or "CVE" in str(e):
        print(f"❌ Hit the torch.load CVE block:\n   {e}")
    else:
        raise
except Exception as e:
    print(f"❌ Unexpected error: {type(e).__name__}: {e}")
