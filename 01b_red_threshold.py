import numpy as np

from cocoatree.io import load_MSA
from cocoatree.statistics.pairwise import compute_sca_matrix
from cocoatree.decomposition import extract_independent_components
from cocoatree.pysca import _icList


def _index_from_nums(nums, target_num: str) -> int:
    try:
        return nums.index(target_num)
    except ValueError as e:
        raise ValueError(f"Could not find residue number {target_num!r} in nums.") from e


def _sector_containing_index(sectors, residue_index: int) -> int:
    containing = [i for i, sector in enumerate(sectors) if residue_index in sector]
    if len(containing) == 0:
        raise ValueError(f"No sector contains residue index {residue_index}.")
    if len(containing) > 1:
        raise ValueError(
            f"Residue index {residue_index} appears in multiple sectors: {containing}"
        )
    return containing[0]


def main() -> None:
    dataset = list(map(np.array, load_MSA("./data/iter_aln_dedup_sp.faa").values()))
    descriptions, sequences = dataset
    weights = np.load("./data/full_weights.npy", allow_pickle=True).ravel()

    rat_trypsin = (
        "IVGGYTCQENSVPYQVSLNS-----GYHFCGGSLINDQWVVSAAHCYKS-------RIQVRLGEHNIN-VLEGNEQFVNAAKIIKHPNFDR--KTLNNDIMLIKLSSPVKLNARVATV-ALPS---SCAP-AG-TQCLISGWGNTLSSG----VNEPDLLQCLDAPLLPQADCEASYP--GKITDNMVCVGFLEGGKDSCQGDSGGPVVCN-----GELQGIVSWGY--GCALPDNPGVYTKVCNYVDWIQDTIAAN---"
    )
    nums = [
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "37",
        "37a",
        "37b",
        "37c",
        "37d",
        "37e",
        "38",
        "39",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "61a",
        "61b",
        "61c",
        "61d",
        "61e",
        "61f",
        "61g",
        "62",
        "63",
        "64",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "74a",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "88",
        "89",
        "90",
        "91",
        "92",
        "93",
        "94",
        "95",
        "96",
        "96a",
        "96b",
        "97",
        "98",
        "99",
        "100",
        "101",
        "102",
        "103",
        "104",
        "105",
        "106",
        "107",
        "108",
        "109",
        "110",
        "111",
        "112",
        "113",
        "114",
        "115",
        "116",
        "117",
        "118",
        "119",
        "120",
        "121",
        "121a",
        "122",
        "123",
        "124",
        "125",
        "125a",
        "125b",
        "125c",
        "127",
        "128",
        "129",
        "130",
        "130a",
        "132",
        "133",
        "133a",
        "134",
        "135",
        "136",
        "137",
        "138",
        "139",
        "140",
        "141",
        "142",
        "143",
        "144",
        "145",
        "146",
        "147",
        "148",
        "148a",
        "148b",
        "148c",
        "148d",
        "149",
        "150",
        "151",
        "152",
        "153",
        "154",
        "155",
        "156",
        "157",
        "158",
        "159",
        "160",
        "161",
        "162",
        "163",
        "164",
        "165",
        "166",
        "167",
        "168",
        "169",
        "170",
        "171",
        "172",
        "173",
        "173a",
        "173b",
        "174",
        "175",
        "176",
        "177",
        "178",
        "179",
        "180",
        "181",
        "182",
        "183",
        "184",
        "184A",
        "185",
        "186",
        "187",
        "188",
        "188A",
        "189",
        "190",
        "191",
        "192",
        "193",
        "194",
        "195",
        "196",
        "197",
        "198",
        "199",
        "200",
        "201",
        "202",
        "202a",
        "202b",
        "202c",
        "202d",
        "202e",
        "203",
        "204",
        "209",
        "210",
        "211",
        "212",
        "213",
        "214",
        "215",
        "216",
        "217",
        "217a",
        "217b",
        "219",
        "220",
        "221",
        "221A",
        "222",
        "223",
        "224",
        "225",
        "226",
        "227",
        "228",
        "229",
        "230",
        "231",
        "232",
        "233",
        "234",
        "235",
        "236",
        "237",
        "238",
        "239",
        "240",
        "241",
        "242",
        "243",
        "244",
        "245",
        "245a",
        "245b",
        "245c",
    ]

    sca_matrix = compute_sca_matrix(sequences, weights)

    # Extract ICs and their cross-correlations, including the statistical cutoffs.
    idpt_components = extract_independent_components(sca_matrix, n_components=3)
    Vica = idpt_components.T  # shape: (n_pos, n_components)
    _, xcor_sizes, sorted_pos, cutoffs, _, _ = _icList(
        Vica, len(idpt_components), sca_matrix
    )

    # Reconstruct the sectors (xcors) exactly as in cocoatree.deconvolution.extract_xcors.
    sectors = [[sorted_pos[i] for i in range(xcor_sizes[0])]]
    ref_index = xcor_sizes[0]
    for isize in range(1, len(xcor_sizes)):
        sectors.append(
            [sorted_pos[i] for i in range(ref_index, ref_index + xcor_sizes[isize])]
        )
        ref_index += xcor_sizes[isize]

    # Identify which sector is the "red" one (the one containing D189 / S189).
    d189_index = _index_from_nums(nums, "189")
    if d189_index >= len(rat_trypsin):
        raise ValueError(
            f"Residue index for nums['189'] is {d189_index}, "
            f"but sequence length is {len(rat_trypsin)}."
        )
    if rat_trypsin[d189_index] != "D":
        raise ValueError(
            f"Expected D at residue number '189' (index {d189_index}), "
            f"found {rat_trypsin[d189_index]!r}."
        )

    red_sector_idx = _sector_containing_index(sectors, d189_index)

    # The threshold used to define that sector is the corresponding cutoff value.
    red_threshold = cutoffs[red_sector_idx]
    print(red_threshold)


if __name__ == "__main__":
    main()

