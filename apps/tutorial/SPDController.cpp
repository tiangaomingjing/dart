#include "SPDController.h"

using namespace Eigen;
using namespace dynamics;

SPDController::SPDController(SkeletonDynamics* _skel, double _timestep) : Controller(_skel) {
    mTimestep = _timestep;

    int nDof = mSkel->getNumDofs();
    mKp = MatrixXd::Identity(nDof, nDof);
    mKd = MatrixXd::Identity(nDof, nDof);
            
    mDesiredDofs = mSkel->getPose();    
    
    mDesiredDofs[9] = -1.5;
    mDesiredDofs[13] = 1.5;
    mDesiredDofs[27] = -1.5;
    mDesiredDofs[28] = 0.75;
    mDesiredDofs[29] = 1.5;
    
    // Using SPD results in simple Kp coefficients
    for (int i = 0; i < 6; i++) {
        mKp(i, i) = 0.0;
        mKd(i, i) = 0.0;
    }
    for (int i = 6; i < 22; i++)
        mKp(i, i) = 400.0; // lower body gain
    for (int i = 22; i < nDof; i++)
        mKp(i, i) = 40.0; // upper body gain
    for (int i = 6; i < 22; i++) 
        mKd(i, i) = 40.0; // lower body damping
    for (int i = 22; i < nDof; i++) 
        mKd(i, i) = 4.0; // upper body damping
}

void SPDController::computeTorques(const VectorXd& _dof, const VectorXd& _dofVel) {    
    
    int nDof = mSkel->getNumDofs();
    MatrixXd invM = (mSkel->getMassMatrix() + mKd * mTimestep).inverse();
    VectorXd p = -mKp * (_dof + _dofVel * mTimestep - mDesiredDofs);
    VectorXd d = -mKd * _dofVel;
    VectorXd qddot = invM * (-mSkel->getCombinedVector() + p + d + mSkel->getConstraintForces());
    mTorques = p + d - mKd * qddot * mTimestep;
    

    // Just to make sure no illegal torque is used    
    for (int i = 0; i < 6; i++)
        mTorques[i] = 0.0;
}
