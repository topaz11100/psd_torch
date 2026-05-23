from __future__ import annotations
import argparse
from psd_snn.artifacts.reader import ArtifactReader
from psd_snn.artifacts.plotting import plot_spectral_curve, plot_spectral_matrix_1d, plot_spectral_matrix_2d, plot_pca_explained_variance, plot_trace_chunk


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--artifact-type', choices=['spectral_curve','spectral_matrix_1d','spectral_matrix_2d','pca_basis','trace'], required=True)
    ap.add_argument('--output', required=True)
    args=ap.parse_args(argv)
    r=ArtifactReader(args.input)
    if args.artifact_type=='spectral_curve': plot_spectral_curve(r,args.output)
    elif args.artifact_type=='spectral_matrix_1d': plot_spectral_matrix_1d(r,args.output)
    elif args.artifact_type=='spectral_matrix_2d': plot_spectral_matrix_2d(r,args.output)
    elif args.artifact_type=='pca_basis': plot_pca_explained_variance(r,args.output)
    else: plot_trace_chunk(r,args.output)

if __name__=='__main__':
    main()
