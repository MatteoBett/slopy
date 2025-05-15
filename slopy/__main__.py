import os
import pickle as pkl

import slopy.loader as loader
import slopy.viz as viz
import slopy.kmers as kmers
import slopy.simcalc as simcalc
import slopy.utils as utils

def main(family_dir : str, path_report : str, path_outdata : str):
    for family_name, batches, k, size_fam in loader.stream_batches(family_dir=family_dir):
        fam_data_out = os.path.join(os.path.join(path_outdata, f"{family_name}"))
        pdf_file = os.path.join(path_report, f"{family_name}.pdf")
        pdf = viz.main_display(batches=batches, path_report=pdf_file)

        batches = kmers.encode_batches(batches=batches, k=k)
        
        for size, distmat in simcalc.stream_batches_similarity(batches=batches):
            simcalc.fill_clusters(distmat=distmat, batch=batches[size])

        
        ordered_batches = sorted(batches, reverse=True)
        batches = {key:batches[key] for key in ordered_batches}
        Laplacian = simcalc.stream_batches_smaller(batches, size_fam=size_fam)
        Laplacian = simcalc.make_laplacian(batches=batches, Laplacian=Laplacian, path_output=fam_data_out)

        viz.degree_v_size(batches=batches, size_fam=size_fam, pdf=pdf)
        viz.show_inter_dist(batches=batches, pdf=pdf, k=k)
        batches, nw_batches = simcalc.reformat(batches=batches)
        
        nw_batches = utils.find_longest_path(batches=nw_batches)        
        #viz.make_heatmap(pdf=pdf, mat=Laplacian, size=size_fam, k=k)
        viz.plot_size_v_lenpath(batches=nw_batches, pdf=pdf)
        viz.slope_length(batches=nw_batches, pdf=pdf)
        #viz.spectral_clustering(Laplacian=Laplacian, batches=batches, pdf=pdf) 
        #viz.show_graph(batches=batches)
        pdf.close()

        with open(os.path.join(path_outdata, f"{family_name}_seqslope.pkl"), "wb") as f:
            pkl.dump({k:seqslope for k, seqslope in nw_batches.items()}, f)

if __name__ == "__main__":
    family_dir = r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/data/test'
    path_report = r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/output/figures'
    path_outdata = r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/output/outdata'
    main(family_dir=family_dir, path_report=path_report, path_outdata=path_outdata)