#include <iostream>
#include <cassert>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "scene.h"   // Assumes scene.h declares the Scene class
#include "mesh.h"    // Assumes mesh.h declares the Mesh class

// Tolerance for floating-point comparisons
const double tol = 1e-6;

// Test: Check that the computed internal force is nearly zero initially.
// (Note: Even if some deformation exists, you can verify that the computed forces are reasonable.)
void testInternalForce(Scene &scene, double dt) {
    Eigen::VectorXd F_internal = -scene.K * (scene.globalPositions - scene.globalOrigPositions);
    std::cout << "[Test] Internal Force: norm(F_internal) = " << F_internal.norm() << std::endl;
    // Expecting a small value for an initially unexcited system
    assert(F_internal.norm() < 1e-3 && "Internal forces are unexpectedly high at initialization");
}

// Test: Verify that each mesh's current positions match their segment in globalPositions.
void testVertexOrderingConsistency(Scene &scene) {
    for (size_t i = 0; i < scene.meshes.size(); ++i) {
        int offset = scene.meshes[i].globalOffset;
        int size = scene.meshes[i].currPositions.size();
        Eigen::VectorXd meshSegment = scene.globalPositions.segment(offset, size);
        double segDiff = (meshSegment - scene.meshes[i].currPositions).norm();
        std::cout << "[Test] Vertex Ordering (mesh " << i << "): norm(diff) = " << segDiff << std::endl;
        assert(segDiff < tol && "Inconsistency in vertex ordering between global and mesh data");
    }
}

// Test: Validate simulation parameters (timeStep, damping α and β).
void testSimulationParameters(double dt, double alpha, double beta) {
    std::cout << "[Test] Simulation Parameters:" << std::endl;
    std::cout << "  timeStep = " << dt << std::endl;
    std::cout << "  alpha = " << alpha << ", beta = " << beta << std::endl;
    assert(dt > 0 && "Time step must be positive");
    assert(alpha >= 0 && alpha <= 0.2 && "Alpha damping parameter out of range [0, 0.2]");
    assert(beta >= 0 && beta <= 0.2 && "Beta damping parameter out of range [0, 0.2]");
}

// Test: Verify tetrahedron volume is as expected.
// For a tetrahedron with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1),
// the volume should be 1/6 ≈ 0.166667.
void testTetVolume(const Mesh &m) {
    if (m.T.rows() > 0) {
        double vol = m.tetVolumes(0);
        std::cout << "[Test] Tetrahedron Volume: " << vol << " (expected ~0.166667)" << std::endl;
        assert(std::abs(vol - 0.166667) < 1e-4 && "Tetrahedron volume is not as expected");
    }
}

// Diagnostic: Print displacement norm, maximum absolute entry in K, and internal force norm.
void diagnosticInternalForce(Scene &scene, double dt) {
    Eigen::VectorXd displacement = scene.globalPositions - scene.globalOrigPositions;
    double dispNorm = displacement.norm();
    
    // Compute maximum absolute value in the stiffness matrix K.
    double maxK = 0;
    for (int k = 0; k < scene.K.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(scene.K, k); it; ++it) {
            maxK = std::max(maxK, std::abs(it.value()));
        }
    }
    
    Eigen::VectorXd F_internal = -scene.K * displacement;
    
    std::cout << "[Diagnostic] Displacement norm = " << dispNorm << std::endl;
    std::cout << "[Diagnostic] Maximum absolute value in K = " << maxK << std::endl;
    std::cout << "[Diagnostic] Internal Force norm = " << F_internal.norm() << std::endl;
}

// Test: Check the condition number and eigenvalues of the system matrix A.
// A = M + dt*D + dt^2*K should be positive definite and well conditioned.
void testMatrixCondition(const Scene &scene) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(scene.A));
    if (es.info() != Eigen::Success) {
        std::cerr << "[Test] Eigenvalue computation failed!" << std::endl;
        return;
    }
    auto eigenvalues = es.eigenvalues();
    double minEig = eigenvalues.minCoeff();
    double maxEig = eigenvalues.maxCoeff();
    std::cout << "[Test] Matrix A Eigenvalues: min = " << minEig << ", max = " << maxEig << std::endl;
    if (minEig < tol) {
        std::cerr << "[Warning] Matrix A is nearly singular." << std::endl;
    }
    std::cout << "[Test] Condition Number ≈ " << maxEig / minEig << std::endl;
}

// Test: Perform one integration step and check for stability.
// The changes in velocity and position should remain within reasonable limits.
void testOneIntegrationStep(Scene &scene, double dt) {
    double initialVelNorm = scene.globalVelocities.norm();
    double initialPosNorm = scene.globalPositions.norm();
    
    // Perform one integration step.
    scene.integrate_global_velocity(dt);
    scene.integrate_global_position(dt);
    
    double newVelNorm = scene.globalVelocities.norm();
    double newPosNorm = scene.globalPositions.norm();
    
    std::cout << "[Test] One Integration Step:" << std::endl;
    std::cout << "  Initial velocity norm = " << initialVelNorm << std::endl;
    std::cout << "  New velocity norm     = " << newVelNorm << std::endl;
    std::cout << "  Initial position norm = " << initialPosNorm << std::endl;
    std::cout << "  New position norm     = " << newPosNorm << std::endl;
    
    // Check that the changes are within a small threshold.
    assert(newVelNorm < 1e-2 && "Velocity magnitude after one step is too high (instability?)");
}

int main() {
    // Create a Scene instance.
    Scene scene;
    
    // Create a simple tetrahedron mesh.
    Eigen::MatrixXd V(4, 3);
    V << 0, 0, 0,
         1, 0, 0,
         0, 1, 0,
         0, 0, 1;
    
    // Boundary faces for the tetrahedron.
    Eigen::MatrixXi F(4, 3);
    F << 0, 1, 2,
         0, 1, 3,
         0, 2, 3,
         1, 2, 3;
    
    // Single tetrahedron connectivity.
    Eigen::MatrixXi T(1, 4);
    T << 0, 1, 2, 3;
    
    // Material parameters.
    double youngModulus = 1e5;
    double poissonRatio = 0.3;
    double density = 1.0;
    bool isFixed = false;
    Eigen::RowVector3d userCOM(0, 0, 0);
    Eigen::RowVector4d userOrientation(1, 0, 0, 0); // Identity quaternion
    
    // Add the mesh to the scene.
    scene.add_mesh(V, F, T, youngModulus, poissonRatio, density, isFixed, userCOM, userOrientation);
    
    // Test the tetrahedron volume for the added mesh.
    testTetVolume(scene.meshes[0]);
    
    // Set simulation parameters.
    double timeStep = 0.01;
    double alpha = 0.1;
    double beta = 0.1;
    
    testSimulationParameters(timeStep, alpha, beta);
    
    // Initialize the scene (this assembles global FEM matrices and factorizes the system matrix).
    scene.init_scene(timeStep, alpha, beta);
    
    // Run our diagnostic for the internal force.
    diagnosticInternalForce(scene, timeStep);
    // Run the tests.
    testInternalForce(scene, timeStep);
    testVertexOrderingConsistency(scene);
    testMatrixCondition(scene);
    testOneIntegrationStep(scene, timeStep);
    
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
