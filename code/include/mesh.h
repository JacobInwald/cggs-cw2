#ifndef MESH_HEADER_FILE
#define MESH_HEADER_FILE

#include <vector>
#include <fstream>
#include "readMESH.h"
#include "auxfunctions.h"
#include "sparse_block_diagonal.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;


//the class the contains each individual rigid objects and their functionality
class Mesh{
public:
    
    //position
    VectorXd origPositions;     //3|V|x1 original vertex positions in xyzxyz format - never change this!
    VectorXd currPositions;     //3|V|x1 current vertex positions in xyzxyz format
    
    //kinematics
    bool isFixed;               //is the object immobile (infinite mass)
    VectorXd currVelocities;    //3|V|x1 velocities per coordinate in xyzxyz format.
    
    double totalInvMass;
    
    MatrixXi T;                 //|T|x4 tetrahdra
    MatrixXi F;                 //|F|x3 boundary faces
    VectorXd invMasses;         //|V|x1 inverse masses of vertices, computed in the beginning as 1.0/(density * vertex voronoi area)
    VectorXd voronoiVolumes;    //|V|x1 the voronoi volume of vertices
    VectorXd tetVolumes;        //|T|x1 tetrahedra volumes
    int globalOffset;           //the global index offset of the of opositions/velocities/impulses from the beginning of the global coordinates array in the containing scene class
    
    VectorXi boundTets;  //just the boundary tets, for collision
    
    double youngModulus, poissonRatio, density, alpha, beta;
    
    SparseMatrix<double> K, M, D;   //The soft-body matrices
    
    //SimplicialLLT<SparseMatrix<double>>* ASolver;   //the solver for the left-hand side matrix constructed for FEM
    
    ~Mesh(){/*if (ASolver!=NULL) delete ASolver;*/}
    
    
    
    bool isNeighborTets(const RowVector4i& tet1, const RowVector4i& tet2){
        for (int i=0;i<4;i++)
            for (int j=0;j<4;j++)
                if (tet1(i)==tet2(j)) //shared vertex
                    return true;
        
        return false;
    }
   

    void compute_mass_matrix(SparseMatrix<double>& M) {
      // Initialize M as a sparse matrix with the correct size
      M.resize(currPositions.size(), currPositions.size());
      
      // Create triplets to build the sparse matrix
      std::vector<Triplet<double>> MTriplets;
      MTriplets.reserve(currPositions.size()); // For diagonal mass matrix
      
      // Create diagonal mass matrix using voronoiVolumes already computed
      for (int i = 0; i < voronoiVolumes.size(); i++) {
          // Calculate mass using density and voronoi volume
          double mass = voronoiVolumes(i) * density;
          
          // If this vertex belongs to a fixed mesh, set mass to effectively infinite
          // (or in practice, zero for the inverse mass)
          if (isFixed) {
              mass = std::numeric_limits<double>::max();
          }
          
          // Add diagonal entries for x, y, z components
          for (int d = 0; d < 3; d++) {
              MTriplets.push_back(Triplet<double>(3*i+d, 3*i+d, mass));
          }
      }
      
      // Build the sparse matrix from triplets
      M.setFromTriplets(MTriplets.begin(), MTriplets.end());
    }

    //--------------------------------------------------------------------
    // Function: create_element_stiffness_matrix
    //
    // Computes the 12x12 element stiffness matrix for a tetrahedron using the Q-matrix design.
    // Input:
    //   tet: a Vector4i containing the indices of the four vertices of the tetrahedron.
    //   origPositions: the global vector of undeformed vertex positions in xyz repeated order.
    //   youngModulus, poissonRatio: material parameters.
    // Output:
    //   Returns a 12x12 stiffness matrix for the element.
    Eigen::Matrix<double, 12, 12> create_element_stiffness_matrix(
        const Eigen::Vector4i &tet,
        const Eigen::VectorXd &origPositions,
        double youngModulus,
        double poissonRatio)
    {
        // 1. Gather the undeformed positions of the tetrahedron vertices.
        Eigen::Matrix<double, 3, 4> X;
        for (int i = 0; i < 4; i++) {
            X.col(i) = origPositions.segment<3>(3 * tet(i));
        }

        // 2. Form the edge matrix: Dm = [X1 - X0, X2 - X0, X3 - X0] and invert.
        Eigen::Matrix3d Dm;
        Dm.col(0) = X.col(1) - X.col(0);
        Dm.col(1) = X.col(2) - X.col(0);
        Dm.col(2) = X.col(3) - X.col(0);
        Eigen::Matrix3d DmInv = Dm.inverse();

        // 3. Compute gradients of shape functions: dN(i) = grad(Ni)
        Eigen::Matrix<double, 3, 4> dN;
        dN.col(0) = - (DmInv.col(0) + DmInv.col(1) + DmInv.col(2));
        dN.col(1) = DmInv.col(0);
        dN.col(2) = DmInv.col(1);
        dN.col(3) = DmInv.col(2);

        // 4. Compute the tetrahedron volume: |det(Dm)|/6.
        double volume = std::abs(Dm.determinant()) / 6.0;

        // 5. Compute Lamé parameters.
        double mu = youngModulus / (2.0 * (1.0 + poissonRatio));
        double lambda = (youngModulus * poissonRatio) / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));

        // 6. Build the constitutive matrix C (6x6) in Voigt notation.
        Eigen::Matrix<double, 6, 6> C = Eigen::Matrix<double, 6, 6>::Zero();
        C(0,0) = lambda + 2.0*mu; C(1,1) = lambda + 2.0*mu; C(2,2) = lambda + 2.0*mu;
        C(0,1) = lambda; C(0,2) = lambda;
        C(1,0) = lambda; C(1,2) = lambda;
        C(2,0) = lambda; C(2,1) = lambda;
        C(3,3) = mu;
        C(4,4) = mu;
        C(5,5) = mu;

        // 7. Build the strain-displacement matrix B (6x12).
        Eigen::Matrix<double, 6, 12> B = Eigen::Matrix<double, 6, 12>::Zero();
        for (int i = 0; i < 4; i++)
        {
            double dNx = dN(0, i);
            double dNy = dN(1, i);
            double dNz = dN(2, i);
            int col = 3 * i;
            // Normal strains.
            B(0, col + 0) = dNx;  // e_xx
            B(1, col + 1) = dNy;  // e_yy
            B(2, col + 2) = dNz;  // e_zz
            // Shear strains.
            B(3, col + 0) = dNy;  // e_xy: du_x/dy
            B(3, col + 1) = dNx;  // e_xy: du_y/dx
            B(4, col + 1) = dNz;  // e_yz: du_y/dz
            B(4, col + 2) = dNy;  // e_yz: du_z/dy
            B(5, col + 2) = dNx;  // e_zx: du_z/dx
            B(5, col + 0) = dNz;  // e_zx: du_x/dz
        }

        // 8. Compute the element stiffness matrix: K_e = B^T * C * B * volume.
        Eigen::Matrix<double, 12, 12> Ke = B.transpose() * (C * B);
        Ke *= volume;

        return Ke;
    }

  // Inside Mesh.h (within the Mesh class definition):

  // Computes the 12x12 element stiffness matrix for a given tetrahedron.
  // Uses the mesh's origPositions, youngModulus, and poissonRatio.
  SparseMatrix<double> create_element_stiffness_matrix(const Eigen::Vector4i &tet) {
      // 1. Gather the undeformed positions of the tetrahedron's vertices.
      Eigen::Matrix<double, 3, 4> X;
      for (int i = 0; i < 4; i++) {
          X.col(i) = origPositions.segment<3>(3 * tet(i));
      }
      
      // 2. Form the edge matrix: Dm = [X1 - X0, X2 - X0, X3 - X0] and compute its inverse.
      Eigen::Matrix3d Dm;
      Dm.col(0) = X.col(1) - X.col(0);
      Dm.col(1) = X.col(2) - X.col(0);
      Dm.col(2) = X.col(3) - X.col(0);
      Eigen::Matrix3d DmInv = Dm.inverse();
      
      // 3. Compute shape function gradients with respect to the rest coordinates.
      Eigen::Matrix<double, 3, 4> dN;
      dN.col(0) = -(DmInv.col(0) + DmInv.col(1) + DmInv.col(2));
      dN.col(1) = DmInv.col(0);
      dN.col(2) = DmInv.col(1);
      dN.col(3) = DmInv.col(2);
      
      // 4. Compute the tetrahedron volume: |det(Dm)|/6.
      double volume = std::abs(Dm.determinant()) / 6.0;
      
      // 5. Compute Lamé parameters.
      double mu = youngModulus / (2.0 * (1.0 + poissonRatio));
      double lambda = (youngModulus * poissonRatio) / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));
      
      // 6. Build the 6x6 constitutive matrix C (Voigt notation).
      Eigen::Matrix<double, 6, 6> C = Eigen::Matrix<double, 6, 6>::Zero();
      C(0,0) = lambda + 2.0*mu;  C(1,1) = lambda + 2.0*mu;  C(2,2) = lambda + 2.0*mu;
      C(0,1) = lambda; C(0,2) = lambda;
      C(1,0) = lambda; C(1,2) = lambda;
      C(2,0) = lambda; C(2,1) = lambda;
      C(3,3) = mu;
      C(4,4) = mu;
      C(5,5) = mu;
      
      // 7. Build the 6x12 strain-displacement matrix B.
      Eigen::Matrix<double, 6, 12> B = Eigen::Matrix<double, 6, 12>::Zero();
      for (int i = 0; i < 4; i++) {
          double dNx = dN(0, i);
          double dNy = dN(1, i);
          double dNz = dN(2, i);
          int col = 3 * i;
          // Normal strains: e_xx, e_yy, e_zz.
          B(0, col + 0) = dNx;
          B(1, col + 1) = dNy;
          B(2, col + 2) = dNz;
          // Shear strains:
          // e_xy = du_x/dy + du_y/dx.
          B(3, col + 0) = dNy;
          B(3, col + 1) = dNx;
          // e_yz = du_y/dz + du_z/dy.
          B(4, col + 1) = dNz;
          B(4, col + 2) = dNy;
          // e_zx = du_z/dx + du_x/dz.
          B(5, col + 2) = dNx;
          B(5, col + 0) = dNz;
      }
      
      // 8. Compute and return the element stiffness matrix: K_e = B^T * C * B * volume.
      Eigen::Matrix<double, 12, 12> Ke = B.transpose() * (C * B);
      Ke *= volume;
      
      return Ke.sparseView();
  }


  // 2. Create the assembly (permutation) matrix Q.
  //    This maps global DOFs to local DOFs.
  //    Q is defined to be of size (12*T x 3*numVerts), where T = number of tetrahedra.
  void create_permutation_matrix(Eigen::SparseMatrix<double> &Q) {
      int T_count = T.rows();                        // number of tetrahedra
      int numVerts = origPositions.size() / 3;         // number of vertices
      int rows = 12 * T_count;                         // local DOF count
      int cols = 3 * numVerts;                         // global DOF count
      
      std::vector<Eigen::Triplet<double>> triplets;
      // Reserve space: each tet contributes 12 entries.
      triplets.reserve(12 * T_count);
      
      // For each tetrahedron e, for each local vertex i, for each coordinate di:
      // localDOF index = 12*e + 3*i + di, and maps to globalDOF = 3*(T(e,i)) + di.
      for (int e = 0; e < T_count; e++) {
          for (int i = 0; i < 4; i++) {
              int vertexIndex = T(e, i);
              for (int di = 0; di < 3; di++) {
                  int localDOF = 12 * e + 3 * i + di;
                  int globalDOF = 3 * vertexIndex + di;
                  triplets.emplace_back(localDOF, globalDOF, 1.0);
              }
          }
      }
      Q.resize(rows, cols);
      Q.setFromTriplets(triplets.begin(), triplets.end());
  }

  // Assembles the global stiffness matrix K using the Q-matrix approach.
  // This function computes each element's stiffness matrix using create_element_stiffness_matrix,
  // builds a block-diagonal Kprime from them, constructs the assembly matrix Q, and finally sets:
  //   K = Q^T * Kprime * Q.
  // The assembled global stiffness matrix is stored in the Mesh attribute K.
  void compute_stiffness_matrix(SparseMatrix<double> &K) {
      int T_count = T.rows();  // number of tetrahedra
      int dimKprime = 12 * T_count;
      
      // 1. Build the block-diagonal matrix Kprime by computing each element's stiffness matrix.
      std::vector<SparseMatrix<double>> localK;

      for (int e = 0; e < T_count; e++) {
          // Get the tetrahedron's vertex indices.
          Eigen::Vector4i tet = T.row(e);
          // Compute the element stiffness matrix.
          SparseMatrix<double> Ke = create_element_stiffness_matrix(tet);
          localK.push_back(Ke);
          
      }
      Eigen::SparseMatrix<double> Kprime(dimKprime, dimKprime);
      
      sparse_block_diagonal(localK, Kprime);

      // 2. Build the assembly (permutation) matrix Q.
      Eigen::SparseMatrix<double> Q;
      create_permutation_matrix(Q);
      
      // 3. Assemble the global stiffness matrix: K = Q^T * Kprime * Q.
      // Compute the product in two steps for clarity.
      Eigen::SparseMatrix<double> A = Kprime * Q;
      K = Q.transpose() * A;
  }

    bool containsNaN(const Eigen::SparseMatrix<double>& mat) {
      for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
            if (std::isnan(it.value())) {
                return true;
            }
        }
      }
      return false;
    }

    bool allNaN(const Eigen::SparseMatrix<double>& mat) {
      for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
            if (!std::isnan(it.value())) {
              return false; 
            }
        }
      }
      return true;
    }


    //Computing the K, M, D matrices per mesh.
    void create_global_matrices(const double timeStep, const double _alpha, const double _beta)
    {
      // K.resize(currVelocities.size(), currVelocities.size());
      compute_stiffness_matrix(K);
      compute_mass_matrix(M); 
      D = _alpha*M + _beta*K;
      cout << containsNaN(M) << " " << containsNaN(K) << " " << containsNaN(D) << endl;
      cout << allNaN(M) << " " << allNaN(K) << " " << allNaN(D) << endl;
    }
    
    //returns center of mass
    Vector3d initializeVolumesAndMasses()
    {
        //TODO: compute tet volumes and allocate to vertices
        tetVolumes.conservativeResize(T.rows());
        voronoiVolumes.conservativeResize(origPositions.size()/3);
        voronoiVolumes.setZero();
        invMasses.conservativeResize(origPositions.size()/3);
        Vector3d COM; COM.setZero();
        for (int i=0;i<T.rows();i++){
            Vector3d e01=origPositions.segment(3*T(i,1),3)-origPositions.segment(3*T(i,0),3);
            Vector3d e02=origPositions.segment(3*T(i,2),3)-origPositions.segment(3*T(i,0),3);
            Vector3d e03=origPositions.segment(3*T(i,3),3)-origPositions.segment(3*T(i,0),3);
            Vector3d tetCentroid=(origPositions.segment(3*T(i,0),3)+origPositions.segment(3*T(i,1),3)+origPositions.segment(3*T(i,2),3)+origPositions.segment(3*T(i,3),3))/4.0;
            tetVolumes(i)=std::abs(e01.dot(e02.cross(e03)))/6.0;
            for (int j=0;j<4;j++)
                voronoiVolumes(T(i,j))+=tetVolumes(i)/4.0;
            
            COM+=tetVolumes(i)*tetCentroid;
        }
        
        COM.array()/=tetVolumes.sum();
        totalInvMass=0.0;
        for (int i=0;i<origPositions.size()/3;i++){
            invMasses(i)=1.0/(voronoiVolumes(i)*density);
            totalInvMass+=voronoiVolumes(i)*density;
        }
        totalInvMass = 1.0/totalInvMass;
        
        return COM;
        
    }
    
    Mesh(const VectorXd& _origPositions, const MatrixXi& boundF, const MatrixXi& _T, const int _globalOffset, const double _youngModulus, const double _poissonRatio, const double _density, const bool _isFixed, const RowVector3d& userCOM, const RowVector4d& userOrientation){
        origPositions=_origPositions;
        //cout<<"original origPositions: "<<origPositions<<endl;
        T=_T;
        F=boundF;
        isFixed=_isFixed;
        globalOffset=_globalOffset;
        density=_density;
        poissonRatio=_poissonRatio;
        youngModulus=_youngModulus;
        currVelocities=VectorXd::Zero(origPositions.rows());
        
        VectorXd naturalCOM=initializeVolumesAndMasses();
        //cout<<"naturalCOM: "<<naturalCOM<<endl;
        
        
        origPositions-= naturalCOM.replicate(origPositions.rows()/3,1);  //removing the natural COM of the OFF file (natural COM is never used again)
        //cout<<"after natrualCOM origPositions: "<<origPositions<<endl;
        
        for (int i=0;i<origPositions.size();i+=3)
            origPositions.segment(i,3) = (QRot(origPositions.segment(i,3).transpose(), userOrientation)+userCOM).transpose();
        
        currPositions=origPositions;
        
        if (isFixed)
            invMasses.setZero();
        
        //finding boundary tets
        VectorXi boundVMask(origPositions.rows()/3);
        boundVMask.setZero();
        for (int i=0;i<boundF.rows();i++)
            for (int j=0;j<3;j++)
                boundVMask(boundF(i,j))=1;
        
        //cout<<"boundVMask.sum(): "<<boundVMask.sum()<<endl;
        
        vector<int> boundTList;
        for (int i=0;i<T.rows();i++){
            int incidence=0;
            for (int j=0;j<4;j++)
                incidence+=boundVMask(T(i,j));
            if (incidence>2)
                boundTList.push_back(i);
        }
        
        boundTets.resize(boundTList.size());
        for (int i=0;i<boundTets.size();i++)
            boundTets(i)=boundTList[i];
        
    }
    
};





#endif
