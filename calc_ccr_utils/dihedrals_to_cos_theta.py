from pyxmolpp2 import TorsionAngle, TorsionAngleFactory, Atom, Degrees, aName
from training_data.ccr_from_dihedrals.calc_ccr_utils.extract_pas import extract_csa_c_x_axis, extract_csa_c_y_axis, \
    extract_csa_c_z_axis
import numpy as np


def cos_angle_between_vectors(v1,
                              v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return np.clip(np.dot(v1, v2), -1.0, 1.0)


def extract_csa_axis(CO_vectors,
                     C_CA_vectors
                     ):
    pas_z = extract_csa_c_z_axis(CO_vectors, C_CA_vectors)
    pas_y = extract_csa_c_y_axis(CO_vectors, pas_z)
    pas_x = extract_csa_c_x_axis(CO_vectors, pas_z)

    return pas_x, pas_y, pas_z


def cos_between_vector_and_csa_axis(CO_vectors,
                                    C_CA_vectors,
                                    vector
                                    ):
    rotated_pas_x, rotated_pas_y, rotated_pas_z = extract_csa_axis(CO_vectors, C_CA_vectors)
    cos_theta_x = cos_angle_between_vectors(rotated_pas_x[0], vector)
    cos_theta_y = cos_angle_between_vectors(rotated_pas_y[0], vector)
    cos_theta_z = cos_angle_between_vectors(rotated_pas_z[0], vector)

    return np.array([cos_theta_x, cos_theta_y, cos_theta_z])


def cos_between_csa_and_csa_axis(CO_vectors_csa_1,
                                 C_CA_vectors_csa_1,
                                 CO_vectors_csa_2,
                                 C_CA_vectors_csa_2
                                 ):
    rotated_pas_x_1, rotated_pas_y_1, rotated_pas_z_1 = extract_csa_axis(CO_vectors_csa_1, C_CA_vectors_csa_1)
    rotated_pas_x_2, rotated_pas_y_2, rotated_pas_z_2 = extract_csa_axis(CO_vectors_csa_2, C_CA_vectors_csa_2)
    cos_theta_x1x2 = cos_angle_between_vectors(rotated_pas_x_1[0], rotated_pas_x_2[0])
    cos_theta_x1y2 = cos_angle_between_vectors(rotated_pas_x_1[0], rotated_pas_y_2[0])
    cos_theta_y1x2 = cos_angle_between_vectors(rotated_pas_y_1[0], rotated_pas_x_2[0])
    cos_theta_y1y2 = cos_angle_between_vectors(rotated_pas_y_1[0], rotated_pas_y_2[0])

    return np.array([cos_theta_x1x2, cos_theta_x1y2, cos_theta_y1x2, cos_theta_y1y2])


def affected_phi_atoms(a: Atom, b: Atom, c: Atom, d: Atom):
    from pyxmolpp2 import rId, aName
    return a.molecule.atoms.filter((rId < a.residue.id) | ((aName == "H") & (rId == a.residue.id)))


def rotatation_psi(r1,
                   psi_degree
                   ):
    # rotation around CA-C residue[i] (all other rotation in i+1)
    # only CO[i] rotates in residue i
    psi = TorsionAngleFactory.psi(r1)
    psi.rotate_to(Degrees(psi_degree))


def rotatation_phi(r1,
                   r2,
                   phi_degree
                   ):
    # rotation around CA-N residue[i]
    # only NH[i] rotates in residue i (all other rotation in i-1)
    phi_2_rw = TorsionAngle(r2["C"], r2["CA"], r2["N"], r1["C"],
                            affected_phi_atoms)
    phi_2_rw.rotate_to(Degrees(phi_degree))


def get_vector(residue,
               name
               ):
    if name == "N_H":
        return residue.atoms.filter(aName == "N").coords.values - residue.atoms.filter(aName == "H").coords.values
    elif name == "CA_HA":
        return residue.atoms.filter(aName == "CA").coords.values - residue.atoms.filter(aName == "HA").coords.values
    elif name == "C_O":
        return residue.atoms.filter(aName == "C").coords.values - residue.atoms.filter(aName == "O").coords.values
    elif name == "C_CA":
        return residue.atoms.filter(aName == "C").coords.values - residue.atoms.filter(aName == "CA").coords.values


def dihedrals_to_cos_theta(ccr_name,
                           tripeptide,
                           phi_degree=0,
                           psi_degree=0):
    r1 = tripeptide.residues[0]
    r2 = tripeptide.residues[1]
    r3 = tripeptide.residues[2]

    if ccr_name == "00_HAHN_C":
        rotatation_psi(r2, psi_degree)
        rotatation_phi(r1, r2, phi_degree)
        HA_HN = r2.atoms.filter(aName == "HA").coords.values - r2.atoms.filter(aName == "H").coords.values
        C_CA = get_vector(r2, "C_CA")
        C_O = get_vector(r2, "C_O")
        return cos_between_vector_and_csa_axis(C_O, C_CA, HA_HN[0])


    elif ccr_name == "01_HAHNp1_C":
        rotatation_psi(r2, psi_degree)
        rotatation_phi(r1, r2, phi_degree)
        HA_HNp1 = r2.atoms.filter(aName == "HA").coords.values - r3.atoms.filter(aName == "H").coords.values
        C_CA = get_vector(r2, "C_CA")
        C_O = get_vector(r2, "C_O")
        return cos_between_vector_and_csa_axis(C_O, C_CA, HA_HNp1[0])

    elif ccr_name == "02_CAHA_Np1Hp1":
        rotatation_psi(r2, psi_degree)
        CA_HA = get_vector(r2, "CA_HA")
        Np1_Hp1 = get_vector(r3, "N_H")
        return cos_angle_between_vectors(CA_HA, Np1_Hp1[0])

    elif ccr_name == "03_CAHA_C":
        rotatation_psi(r2, psi_degree)
        CA_HA = get_vector(r2, "CA_HA")
        C_CA = get_vector(r2, "C_CA")
        C_O = get_vector(r2, "C_O")
        return cos_between_vector_and_csa_axis(C_O, C_CA, CA_HA[0])

    elif ccr_name == "04_CAHA_NH":
        rotatation_phi(r1, r2, phi_degree)
        CA_HA = get_vector(r2, "CA_HA")
        N_H = get_vector(r2, "N_H")
        return cos_angle_between_vectors(CA_HA, N_H[0])

    elif ccr_name == "05_CAHA_Cm1":
        rotatation_phi(r1, r2, phi_degree)
        CA_HA = get_vector(r2, "CA_HA")
        C_CA = get_vector(r1, "C_CA")
        C_O = get_vector(r1, "C_O")
        return cos_between_vector_and_csa_axis(C_O, C_CA, CA_HA[0])

    elif ccr_name == "06_NH_C":
        rotatation_psi(r2, psi_degree)
        rotatation_phi(r1, r2, phi_degree)
        N_H = get_vector(r2, "N_H")
        C_CA = get_vector(r2, "C_CA")
        C_O = get_vector(r2, "C_O")
        return cos_between_vector_and_csa_axis(C_O, C_CA, N_H[0])

    elif ccr_name == "07_Cm1_C":
        rotatation_psi(r2, psi_degree)
        rotatation_phi(r1, r2, phi_degree)
        C_CAm1 = get_vector(r1, "C_CA")
        C_Om1 = get_vector(r1, "C_O")
        C_CA = get_vector(r2, "C_CA")
        C_O = get_vector(r2, "C_O")
        return cos_between_csa_and_csa_axis(C_Om1, C_CAm1, C_O, C_CA)

    elif ccr_name == "08_NH_Np1Hp1":
        rotatation_psi(r2, psi_degree)
        rotatation_phi(r1, r2, phi_degree)
        N_H = get_vector(r2, "N_H")
        Np1_Hp1 = get_vector(r3, "N_H")
        return cos_angle_between_vectors(N_H, Np1_Hp1[0])


def dihedrals_to_distance(ccr_name,
                          tripeptide,
                          phi_degree=0,
                          psi_degree=0
                          ):
    r1 = tripeptide.residues[0]
    r2 = tripeptide.residues[1]
    r3 = tripeptide.residues[2]
    if ccr_name == "00_HAHN_C":
        rotatation_psi(r2, psi_degree)
        rotatation_phi(r1, r2, phi_degree)
        HA_HN = r2.atoms.filter(aName == "HA").coords.values - r2.atoms.filter(aName == "H").coords.values
        return np.linalg.norm(HA_HN)

    elif ccr_name == "01_HAHNp1_C":
        rotatation_psi(r2, psi_degree)
        rotatation_phi(r1, r2, phi_degree)
        HA_HNp1 = r2.atoms.filter(aName == "HA").coords.values - r3.atoms.filter(aName == "H").coords.values
        return np.linalg.norm(HA_HNp1)
