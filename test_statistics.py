# Project authors: Laura Maria Fetz, Martin Turna, and Bart Amin
"""Small deterministic checks of pairing and exact discordant-pair tests."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from analysis import summarize

def test_identical_models_align_by_question():
    frame=pd.DataFrame({"Question":[f"Q{i}" for i in range(100)],"Type":"Adversarial", "Is_Correct":np.arange(100)%2,"Cosine_Similarity":.7})
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)
        summarize({"A":frame,"B":frame.iloc[::-1]},path,1000)
        pair=pd.read_csv(path/"paired_comparisons.csv").iloc[0]
        assert pair.Difference_A_minus_B==0
        assert pair.CI_Lower==0 and pair.CI_Upper==0
        assert pair.McNemar_exact_p==1 and pair.Holm_adjusted_p==1

def test_exact_mcnemar_and_empty_filtered_subset():
    frame=pd.DataFrame({"Question":[f"Q{i}" for i in range(100)],"Type":"Adversarial", "Is_Correct":np.zeros(100,dtype=int),"Cosine_Similarity":.2})
    improved=frame.copy(); improved.loc[:9,"Is_Correct"]=1
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)
        summarize({"A":frame,"B":improved},path,1000)
        pair=pd.read_csv(path/"paired_comparisons.csv").iloc[0]
        summary=pd.read_csv(path/"model_summary.csv")
        assert np.isclose(pair.Difference_A_minus_B,-.1)
        assert pair.A_only_correct==0 and pair.B_only_correct==10
        assert np.isclose(pair.McNemar_exact_p,2*(.5**10))
        assert summary.Correctness_Filtered.isna().all()

if __name__=="__main__":
    test_identical_models_align_by_question()
    test_exact_mcnemar_and_empty_filtered_subset()
    print("Passed: question alignment, zero discordance, exact paired p-value, empty filtered subset")
