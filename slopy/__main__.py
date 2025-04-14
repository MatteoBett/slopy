import os, re

import slopy.loader as loader
import slopy.viz as viz
import slopy.kmers as kmers
import slopy.simcalc as simcalc

def main(family_dir : str, path_report : str):
    for family_name, batches, k in loader.stream_batches(family_dir=family_dir):
        pdf_file = os.path.join(path_report, f"{family_name}.pdf")
        pdf = viz.main_display(batches=batches, path_report=pdf_file)

        batches = kmers.encode_batches(batches=batches, k=k)

        for size, distmat in simcalc.stream_batches_similarity(batches=batches):
            simcalc.fill_clusters(distmat=distmat, batch=batches[size])
            viz.make_heatmap(pdf=pdf, distmat=distmat, size=size, k=k)

        ordered_batches = sorted(batches, reverse=True)
        batches = {key:batches[key] for key in ordered_batches}

        simcalc.stream_batches_smaller(batches)
        viz.show_inter_dist(batches=batches, pdf=pdf, k=k)
        pdf.close()
        viz.show_graph(batches=batches)

if __name__ == "__main__":
    family_dir = r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/data/test'
    path_report = r'/home/mbettiati/LBE_MatteoBettiati/code/slopy/output/figures'
    main(family_dir=family_dir, path_report=path_report)