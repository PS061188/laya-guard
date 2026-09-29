import json, sys
for tag in sys.argv[1:]:
    r = json.load(open(f"data/finetune_results_{tag}.json"))
    for when in (("before", "after") if tag == sys.argv[1] else ("after",)):
        x = r[when]
        f = lambda v: "-" if v is None else f"{v*100:.0f}%"
        for mode in ("alone", "with_rules"):
            lt, rt = x[mode]["lab_test"], x[mode]["real_test"]
            print(f"{tag:5s} {when:6s} {mode:10s} t={x['thresholds'][mode]:.3f} LAB fa {f(lt['false_alarms'])} miss {f(lt['misses'])} auc {lt['auroc']:.2f} | REAL fa {f(rt['false_alarms'])} miss {f(rt['misses'])} ({rt['n_destructive']}) pause {f(rt['pause_rate'])} auc {rt['auroc']:.2f} | unseen {x[mode]['real_unseen_pause_rate']*100:.1f}%")
        print("      length", [(b['chars'], round(b['mean_score'], 2), f"{b['flag_rate']*100:.0f}%") for b in x["length_real_test_harmless"]])
