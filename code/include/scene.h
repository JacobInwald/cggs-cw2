#ifndef SCENE_HEADER_FILE
#define SCENE_HEADER_FILE

#include <vector>
#include <queue>
#include <fstream>
#include "auxfunctions.h"
#include "readMESH.h"
#include "sparse_block_diagonal.h"
#include "mesh.h"
#include "sparse_block.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>

using namespace Eigen;
using namespace std;

class Scene{
public:
    double timeStep;
    double currTime;
    bool recomputeSolver;
    
    VectorXd globalOrigPositions;
    VectorXd globalPositions;   // 3*|V| all positions
    VectorXd globalVelocities;  // 3*|V| all velocities
    VectorXd globalInvMasses;   // 3*|V| all inverse masses (NOTE: the invMasses in the Mesh class is |v| (one per vertex)!)
    MatrixXi globalT;           // |T| x 4 tetrahedra in global index
    
    SparseMatrix<double> K, M, D, A;
    SparseLU<SparseMatrix<double>> ASolver;
    
    vector<Mesh> meshes;

    // updates from global values back into mesh values
    void global2Mesh(){
        for (int i=0; i<meshes.size(); i++){
            if (meshes[i].isFixed)
                continue;
            meshes[i].currPositions = globalPositions.segment(meshes[i].globalOffset, meshes[i].currPositions.size());
            meshes[i].currVelocities = globalVelocities.segment(meshes[i].globalOffset, meshes[i].currVelocities.size());
        }
    }
    
    // update from mesh current values into global values
    void mesh2global(){
        for (int i=0; i<meshes.size(); i++){
            if (meshes[i].isFixed)
                continue;
            globalPositions.segment(meshes[i].globalOffset, meshes[i].currPositions.size()) = meshes[i].currPositions;
            globalVelocities.segment(meshes[i].globalOffset, meshes[i].currVelocities.size()) = meshes[i].currVelocities;
        }
    }
    
    
    // This should be called whenever the timestep changes.
    // Here we compute the per-mesh FEM matrices and assemble the full-scene matrices,
    // then pre-factorize the left-hand side A = M + timeStep*D + (timeStep^2)*K using ASolver.
    void init_scene(double _timeStep, const double alpha, const double beta){
       // Update the time step
        timeStep = _timeStep;
       
        // Create block diagonal matrices
        std::vector<SparseMatrix<double>> blockK, blockM, blockD;

        // Compute per-mesh matrices
        for (int i=0; i<meshes.size(); i++){
            if (meshes[i].isFixed)
              continue;
            // Compute local matrices
            meshes[i].create_global_matrices(timeStep, alpha, beta);
            // Extract local matrices as blocks
            blockK.push_back(meshes[i].K);
            blockM.push_back(meshes[i].M);
            blockD.push_back(meshes[i].D);
        }
        
        // Use block diagonal method to assemble global matrices
        sparse_block_diagonal(blockK, K);
        sparse_block_diagonal(blockM, M);
        sparse_block_diagonal(blockD, D);
        

        // Create left-hand side matrix A = M + timeStep*D + (timeStep^2)*K
        A = M + timeStep * D + timeStep * timeStep * K;
        
        // Pre-factorize the left-hand side matrix
        ASolver.analyzePattern(A);
        ASolver.factorize(A);
        
        // Check if the solver succeeded
        if (ASolver.info() != Eigen::Success) {
            std::cerr << "Solver factorization failed!" << std::endl;
        }
    }
    
    
    // Performs the integration step for global velocities.
    // Implicit integration based on:
    // (M + timeStep*D + (timeStep^2)*K) * v(t+timeStep) = M*v(t) - timeStep*K*(x(t)-x(0))
    // Here we assume no external force (f_e = 0).
    void integrate_global_velocity(double timeStep){
        // Compute the right-hand side: M * current velocities - timeStep * K * (currentPositions - originalPositions)
        VectorXd rhs = M * globalVelocities - timeStep * (K * (globalPositions - globalOrigPositions));
        
        // Solve for the new velocities.
        VectorXd newVelocities = ASolver.solve(rhs);
        
        // Update global velocities.
        globalVelocities = newVelocities;
    }
    
    // Update the global positions using the newly computed velocities.
    // x(t+timeStep) = x(t) + timeStep * v(t+timeStep)
    void integrate_global_position(double timeStep){
        globalPositions = globalPositions + timeStep * globalVelocities;
    }
    
    
    // Update the scene: integrate velocity, update positions, and copy back to meshes.
    void update_scene(double timeStep){
        integrate_global_velocity(timeStep);
        integrate_global_position(timeStep);
        global2Mesh();
    }
    
    // Adding an object.
    void add_mesh(const MatrixXd& V, const MatrixXi& boundF, const MatrixXi& T, 
                  const double youngModulus, const double PoissonRatio,  
                  const double density, const bool isFixed, 
                  const RowVector3d& userCOM, const RowVector4d userOrientation){
        
        VectorXd Vxyz(3*V.rows());
        for (int i=0; i<V.rows(); i++)
            Vxyz.segment(3*i,3) = V.row(i).transpose();
        
        Mesh m(Vxyz, boundF, T, globalPositions.size(), youngModulus, PoissonRatio, density, isFixed, userCOM, userOrientation);
        meshes.push_back(m);
        int oldTsize = globalT.rows();
        globalT.conservativeResize(globalT.rows() + T.rows(), 4);
        globalT.block(oldTsize, 0, T.rows(), 4) = T.array() + globalPositions.size()/3;  // offset T to global index
        globalOrigPositions.conservativeResize(globalOrigPositions.size() + Vxyz.size());
        globalPositions.conservativeResize(globalPositions.size() + Vxyz.size());
        globalVelocities.conservativeResize(globalPositions.size());
        int oldIMsize = globalInvMasses.size();
        globalInvMasses.conservativeResize(globalPositions.size());
        for (int i=0; i<m.invMasses.size(); i++)
            globalInvMasses.segment(oldIMsize + 3*i, 3) = Vector3d::Constant(m.invMasses(i));
        
        mesh2global();
    }
    
    // Loading a scene from the scene .txt files.
    // You do not need to update this function.
    void load_scene(const std::string sceneFileName){
        ifstream sceneFileHandle, constraintFileHandle;
        sceneFileHandle.open(DATA_PATH "/" + sceneFileName);
        assert(sceneFileHandle.is_open() && "couldn't read scene file!");
        int numofObjects, numofImpulses;
        
        currTime = 0;
        sceneFileHandle >> numofObjects;
        for (int i = 0; i < numofObjects; i++){
            MatrixXi objT, objF;
            MatrixXd objV;
            std::string MESHFileName;
            bool isFixed;
            double youngModulus, poissonRatio, density;
            RowVector3d userCOM;
            RowVector4d userOrientation;
            sceneFileHandle >> MESHFileName >> density >> youngModulus >> poissonRatio >> isFixed 
                            >> userCOM(0) >> userCOM(1) >> userCOM(2)
                            >> userOrientation(0) >> userOrientation(1) >> userOrientation(2) >> userOrientation(3);
            userOrientation.normalize();
            readMESH(DATA_PATH + std::string("/") + MESHFileName, objV, objF, objT);
            
            // Fixing weird orientation problem
            MatrixXi tempF(objF.rows(), 3);
            tempF << objF.col(2), objF.col(1), objF.col(0);
            objF = tempF;
            
            add_mesh(objV, objF, objT, youngModulus, poissonRatio, density, isFixed, userCOM, userOrientation);
        }
        
        sceneFileHandle >> numofImpulses;
        for (int i = 0; i < numofImpulses; i++){
            int meshNum, vertexNum;
            RowVector3d velocity;
            sceneFileHandle >> meshNum >> vertexNum >> velocity(0) >> velocity(1) >> velocity(2);
            for (int j = 0; j < 3; j++)
                meshes[meshNum].currVelocities[3*vertexNum + j] = velocity(j);
        }
        
        mesh2global();
    }
    
    Scene(){}
    ~Scene(){}
};

#endif
