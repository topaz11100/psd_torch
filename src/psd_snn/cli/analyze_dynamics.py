from psd_snn.analysis.dynamics.runner import analyze_dynamics

if __name__ == '__main__':
    out = analyze_dynamics({'n_params':0.0}, {'spike_rate':0.0})
    print(out)
