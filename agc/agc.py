#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""OTU clustering"""

import argparse
import sys
import gzip
import textwrap
from pathlib import Path
from typing import Iterator, List
# https://github.com/briney/nwalign3
# ftp://ftp.ncbi.nih.gov/blast/matrices/
import nwalign3 as nw
import numpy as np
np.int = int

__author__ = "Shahina MOHAMED"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Shahina"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shahina"
__email__ = "your@email.fr"
__status__ = "Developpement"



def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments(): # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=f"{sys.argv[0]} -h")
    parser.add_argument('-i', '-amplicon_file', dest='amplicon_file', type=isfile, required=True,
                        help="Amplicon is a compressed fasta file (.fasta.gz)")
    parser.add_argument('-s', '-minseqlen', dest='minseqlen', type=int, default = 400,
                        help="Minimum sequence length for dereplication (default 400)")
    parser.add_argument('-m', '-mincount', dest='mincount', type=int, default = 10,
                        help="Minimum count for dereplication  (default 10)")
    parser.add_argument('-o', '-output_file', dest='output_file', type=Path,
                        default=Path("OTU.fasta"), help="Output file")
    return parser.parse_args()


def read_fasta(amplicon_file: Path, minseqlen: int) -> Iterator[str]:
    """Read a compressed fasta and extract sequences of length >= minseqlen.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length.
    :return: A generator object that provides the Fasta sequences (str).
    """
    with gzip.open(amplicon_file, 'rt') as file_gz:
        sequence = ""
        for line in file_gz:
            line = line.strip()
            if line.startswith(">"):
                # si assez long on fait le yield
                if len(sequence) >= minseqlen:
                    yield sequence
                sequence = ""
            else:
                # concatenation
                sequence += line
        # on regarde la dernière seq
        if len(sequence) >= minseqlen:
            yield sequence


def dereplication_fulllength(amplicon_file: Path, minseqlen: int, mincount: int) -> Iterator[List]:
    """Dereplicate the set of sequences based on length and count criteria.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length.
    :param mincount: (int) Minimum amplicon count.
    :return: A generator object that provides a list [sequences, count] of sequences
             with a count >= mincount and a length >= minseqlen.
    """

    sequence_counts = {}
    for sequence in read_fasta(amplicon_file, minseqlen):
        if sequence in sequence_counts:
            sequence_counts[sequence] += 1
        else:
            sequence_counts[sequence] = 1

    # Filtrer sequences basée sur count et longueur
    filtered_sequences = [(sequence, count) for sequence, count in sequence_counts.items()
                          if count >= mincount]
    # triez seq dans ordre decroissant
    sorted_sequences = sorted(filtered_sequences, key=lambda x: x[1], reverse=True)
    # Yield sequences et counts
    for sequence, count in sorted_sequences:
        yield [sequence, count]


def get_identity(alignment_list: List[str]) -> float:
    """Compute the identity rate between two sequences

    :param alignment_list:  (list) A list of aligned sequences 
        in the format ["SE-QUENCE1", "SE-QUENCE2"]
    :return: (float) The rate of identity between the two sequences.
    """
    if len(alignment_list) != 2:
        raise ValueError("Alignment list must contain exactly two sequences.")

    sequence1, sequence2 = alignment_list

    if len(sequence1) != len(sequence2):
        raise ValueError("Sequences in the alignment list must have the same length.")

    identical_count = sum(1 for a, b in zip(sequence1, sequence2) if a == b)
    alignment_length = len(sequence1)

    identity_percentage = (identical_count / alignment_length) * 100.0
    return identity_percentage



def abundance_greedy_clustering(amplicon_file: Path, minseqlen: int,
                                mincount: int, chunk_size: int, kmer_size: int) -> List:
    """Compute an abundance greedy clustering regarding sequence count and identity.
    Identify OTU sequences.

    :param amplicon_file: (Path) Path to the amplicon file in FASTA.gz format.
    :param minseqlen: (int) Minimum amplicon sequence length.
    :param mincount: (int) Minimum amplicon count.
    :param chunk_size: (int) A fournir mais non utilise cette annee
    :param kmer_size: (int) A fournir mais non utilise cette annee
    :return: (list) A list of all the [OTU (str), count (int)] .
    """
    # Appel à dereplication_fulllength pour obtenir les séquences et leurs comptages
    sequences = dereplication_fulllength(amplicon_file, minseqlen, mincount)

    # Initialisation de la liste des OTUs
    otu_list = []

    # Parcours des séquences
    for sequence, count in sequences:
        is_otu = True  # Variable pour vérifier si la séquence est une OTU

        # Comparaison avec les OTUs existantes
        for otu, _ in otu_list:
            # Alignement global
            alignment = nw.global_align(sequence, otu, gap_open=-1, gap_extend=-1,
                                        matrix=str(Path(__file__).parent / "MATCH"))
            # Calcul de l'identité
            identity = get_identity(alignment)

            # Si l'identité est > 97%, alors la séquence n'est pas une OTU
            if identity > 97:
                is_otu = False
                break  # Pas besoin de vérifier les autres OTUs

        # Si la séquence est une OTU, ajout à la liste des OTUs
        if is_otu:
            otu_list.append([sequence, count])
    return otu_list


def write_OTU(otu_list: List, output_file: Path) -> None:
    """Write the OTU sequence in fasta format.

    :param OTU_list: (list) A list of OTU sequences
    :param output_file: (Path) Path to the output file
    """
    with output_file.open("w") as file:
        for idx, (sequence, occurrence) in enumerate(otu_list, start=1):
            header = f">OTU_{idx} occurrence:{occurrence}"
            formatted_sequence = textwrap.fill(sequence, width=80)
            file.write(f"{header}\n{formatted_sequence}\n")

#==============================================================
# Main program
#==============================================================
def main(): # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()
    # Votre programme ici

    # abondance
    chunk_size = 0  # non utilisé
    kmer_size = 0   # non utilisé
    otu_list = abundance_greedy_clustering(
        args.amplicon_file, args.minseqlen, args.mincount, chunk_size, kmer_size)

    # # Affichage des OTUs et de leurs comptages
    # for otu in otu_list:
    #     sequence, count = otu
    #     print(f"OTU: {sequence}")
    #     print(f"Count: {count}")

    # write_OTU
    write_OTU(otu_list, args.output_file)


if __name__ == '__main__':
    main()
