"""Roda todas as configurações x algoritmos x 10 sementes (retomável)."""
import sys, time
sys.path.insert(0, '../src')
import runner
cfgs = ['original', 'M1_f1', 'M2_bacc_bal', 'M3_f1_smote', 'M2b_bacc', 'M3b_acc_smote', 'M4_mcc', 'X1_acc_sem24', 'X2_f1_sem24']
t = time.time()
df = runner.run_many(cfgs, runner.ALGOS, list(range(10)), n_jobs=8)
print('FIM', len(df), 'execuções em', round((time.time() - t) / 60), 'min')
