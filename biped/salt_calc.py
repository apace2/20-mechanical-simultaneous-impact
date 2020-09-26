import numpy as np
import scipy.linalg as la

from .Biped import PCrBiped, DecoupledBiped

def salt_biped(rho, drho, biped, p):
    '''
    Calculate saltation matrix for biped with
    drho    - preimpact velocity
    rho     - impact configuration
    biped   - the biped model
    p       - paramters

    Assumes simultaneous impact configuration

    # parameters
    # mb, mf, Ib - mass of body, mass of foot, intertia of body
    # g          - gravity constant
    # k, l       - sping constant, nominal spring length
    # b          - force associated with flywheel constant
    '''

    # Assume the reset map is independent of time
    t = k = 0

    d = len(rho)
    Jpre = [0, 0]
    Ja1 = [1, 0]
    Ja2 = [0, 1]
    Ja12 = [1, 1]
    # Reset maps - for both PCr and decoupled biped, Delta independent of dq
    Delta1 = biped.Delta(t, k, rho, drho, Ja1, p)
    Delta2 = biped.Delta(t, k, rho, drho, Ja2, p)
    Delta12 = biped.Delta(t, k, rho, drho, Ja12, p)
    # Derivate of reset maps
    # As the mass matrix is constant, DR = diag( I, \Delta(q, \dot{q}))
    # DR is the same for both biped models
    DRa1 = la.block_diag(np.eye(d), Delta1)
    DRa2 = la.block_diag(np.eye(d), Delta2)
    DRa12 = la.block_diag(np.eye(d), Delta12)

    # post impact velocities
    drho1 = Delta1@drho  # velocity after constraint a1 is activated
    drho2 = Delta2@drho
    drho12 = Delta12@drho

    # accelerations
    ddq = biped.ddq(t, k, rho, drho, Jpre, p)
    ddq1 = biped.ddq(t, k, rho, drho1, Ja1, p)
    ddq2 = biped.ddq(t, k, rho, drho2, Ja2, p)
    ddq12 = biped.ddq(t, k, rho, drho12, Ja12, p)

    # vector fields
    V0 = np.hstack([drho, ddq])
    V1 = np.hstack([drho1, ddq1])
    V2 = np.hstack([drho2, ddq2])
    V12 = np.hstack([drho12, ddq12])

    # constraints
    Da1 = biped.Da(t, k, rho, Ja1, p)
    Da2 = biped.Da(t, k, rho, Ja2, p)
    Da12 = biped.Da(t, k, rho, Ja12, p)

    # salt = DR + (F^+ - DR F^-) Dh / Dh F^-
    def salt(DR, Fplus, Fminus, Dh):
        return DR + np.outer((Fplus - DR@Fminus), Dh) / (Dh.dot(Fminus))

    def Dh(Da):
        return np.hstack([Da, np.zeros_like(Da)])

    #calculate saltation matrices
    S1_0 = salt(DRa1, V1, V0, Dh(Da1))  #saltation associated with impacting constraint 1
                                        #from unconstrained
    S2_0 = salt(DRa2, V2, V0, Dh(Da2))  #saltation associated with impacting constraint 2
    S2_1 = salt(DRa2, V12, V1, Dh(Da2))  #saltation impacting constraint 2 after constraint 1
    S1_2 = salt(DRa1, V12, V2, Dh(Da1))

    #saltation matrix corresponding to sim. impact with word w={empty, {1}, {1,2}}
    Xi_21 = S2_1@S1_0
    #saltation matrix corresponding to sim. impact with word w={empty, {2}, {1,2}}
    Xi_12 = S1_2@S2_0

    return Xi_21, Xi_12



if __name__ == '__main__':
    C1B = DecoupledBiped  #at least C1 systems
    PCrB = PCrBiped  #PCr biped
    p = PCrB.nominal_parameters()

    q0, dq0, J = PCrB.ic(p)
    dq0[[PCrB.izb, PCrB.izl, PCrB.izr]] = -1

    #test the saltation matrix calculation, computation does not depend an a(q) = 0
    Xi_21, Xi_12 = salt_biped(q0, dq0, C1B, p)

    # for the C1 case, both saltation matrices must be the same
    assert np.all(Xi_21 == Xi_12)

    # for the PCr case, the saltation matrices may differ
    Xi_21, Xi_12 = salt_biped(q0, dq0, PCrB, p)
