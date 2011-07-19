/*
  RTQL8, Copyright (c) 2011 Georgia Tech Graphics Lab
  All rights reserved.

  Author	Sehoon Ha
  Date		06/12/2011
*/

#include "TrfmRotateQuat.h"
#include "utils/RotationConversion.h"

#include <cassert>
using namespace std;

#include "Dof.h"

namespace model3d {

    TrfmRotateQuat::TrfmRotateQuat(Dof *w, Dof *x, Dof *y, Dof *z, const char *_name){
        mDofs.clear();
        mDofs.resize(4);
        mDofs[0]=w;
        mDofs[1]=x;
        mDofs[2]=y;
        mDofs[3]=z;
        mType = Transformation::T_ROTATEQUAT;
        if(_name != NULL)
            strcpy(mName, _name);
        else
            strcpy(mName, "QUAT");
    }

    void TrfmRotateQuat::evalTransform(){
        // Quaternion constructor takes (w, x, y, z)
        Quaterniond q(mDofs[0]->getValue(), mDofs[1]->getValue(),mDofs[2]->getValue(),mDofs[3]->getValue());
        q.normalize();

        mTransform.setZero();
        Matrix3d rot = q.matrix();
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                mTransform(i, j) = rot(i, j);
            }
        }
        mTransform(3, 3) = 1.0;
    }

    Matrix4d TrfmRotateQuat::getDeriv(const Dof *d){
        Quaterniond q(mDofs[0]->getValue(), mDofs[1]->getValue(), mDofs[2]->getValue(), mDofs[3]->getValue());
        q.normalize();	

        Matrix3d mat = Matrix3d::Zero();
        Matrix4d ret = Matrix4d::Zero();

        int el=-1;
        for(int i=0; i<4; i++) if(d==mDofs[i]) el=i;
        assert(el!=-1);
        // TODO need to rewrite
        //mat = utils::rot_conv::getDerivativeMatrix(q, el);
	
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++)
                ret(i, j) = mat(i, j);
        }

        return ret;
    }

    Matrix4d TrfmRotateQuat::getSecondDeriv(const Dof *d1, const Dof *d2){
        Quaterniond q(mDofs[0]->getValue(), mDofs[1]->getValue(), mDofs[2]->getValue(), mDofs[3]->getValue());
        q.normalize();
	
        Matrix3d mat = Matrix3d::Zero();
        Matrix4d ret = Matrix4d::Ones();

        int el1=-1, el2=-1;
        for(int i=0; i<4; i++) {
            if(d1==mDofs[i]) el1=i;
            if(d2==mDofs[i]) el2=i;
        }
        assert(el1!=-1);
        assert(el2!=-1);
        // TODO need to rewrite
        //mat = utils::rot_conv::getDerivativeMatrix(q, el1, el2);
	
        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++)
                ret(i, j) = mat(i, j);
        }
        return ret;
    }

    void TrfmRotateQuat::applyGLTransform(renderer::RenderInterface* _ri) const
    {
#if 0	
        Quaterniond q(mDofs[0]->getValue(), mDofs[1]->getValue(), mDofs[2]->getValue(), mDofs[3]->getValue());
        q.normalize();
/*
  mDofs[0]->setValue(q.w);
  mDofs[1]->setValue(q.x);
  mDofs[2]->setValue(q.y);
  mDofs[3]->setValue(q.z);
*/
        AngleAxisd angleAxis(q);
        Vector3d a = angleAxis.axis();
        double theta = angleAxis.angle();
        if(a.dot(a)>EPSILON*EPSILON && theta>EPSILON)
        {
#if 1
            if (_ri)
                _ri->Rotate(a, theta*180/M_PI);
#else
            glRotatef(theta*180/M_PI, a(0), a(1), a(2));
#endif
        }
#endif
    }
} // namespace model3d
