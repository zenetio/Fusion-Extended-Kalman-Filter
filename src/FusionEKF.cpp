#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  
  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // measurement matrix - laser
  H_laser_ << 1., 0, 0, 0,
              0, 1., 0, 0;

  /**
   * Initializing the FusionEKF.
   * Set the process and measurement noises
   */
    // create a 4D state vector, we don't know yet the values of the x state
    VectorXd x = VectorXd(4);
    x << 1., 1., 1., 1.;
    // state covariance matrix P
    MatrixXd P = MatrixXd(4, 4);
    P << 1., 0, 0, 0,
        0, 1., 0, 0,
        0, 0, 1000., 0,
        0, 0, 0, 1000.;

    // measurement covariance
    MatrixXd R = MatrixXd(2, 2);
    R << 0.0225, 0,
        0, 0.0225;

    // measurement matrix
    MatrixXd H = MatrixXd(2, 4);
    H << 1., 0, 0, 0,
        0, 1., 0, 0;

    // the initial transition matrix F_
    MatrixXd F = MatrixXd(4, 4);
    F << 1., 0, 1., 0,
        0, 1., 0, 1.,
        0, 0, 1., 0,
        0, 0, 0, 1.;

    MatrixXd Q = MatrixXd(4,4);

    // call Init()
    ekf_.Init(x, P, F, H, R, Q);
  
    // acceleration noise parameters
    noise_ax = 9.0;
    noise_ay = 9.0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

/*--------------------------------------------------------------------------------------------------
 * ProcessMeasurement
 * Process the measurements provided by RADAR and LASER sensors.
 * Here, EKF generates the fusion of both sensor measures.
 * -------------------------------------------------------------------------------------------------
 */
void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
    if (!is_initialized_) {
        /**
         * Initialize the state ekf_.x_ with the first measurement.
         * Create the covariance matrix.
         * Need to convert radar from polar to cartesian coordinates.
         */

        // first measurement
        cout << "Initializing EKF... " << endl;
        
        previous_timestamp_ = 0;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            // Convert radar from polar to cartesian coordinates 
            // and initialize state.
            double ro = measurement_pack.raw_measurements_[0];
            double theta = measurement_pack.raw_measurements_[1];
            double ro_dot= measurement_pack.raw_measurements_[2];
            // must convert from polar to cartesian
            // or
            // ro, theta, ro_dot -> x, y, vx, vy
            double x = ro * cos(theta);
            if(x < 0.0001) x = 0.0001;
            double y = ro * sin(theta);
            if(y < 0.0001) y = 0.0001;
            double px = ro_dot*cos(theta);
            double py = ro_dot*sin(theta);
            ekf_.x_  << x, y, px, py;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            // Initialize state.
            ekf_.x_ << measurement_pack.raw_measurements_[0],   // px
                   measurement_pack.raw_measurements_[1],       // py
                    0., 0.;
        }
        // get the elapsed time
        previous_timestamp_ = measurement_pack.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /**----------------------------------------------------
     * Prediction operation
     *-----------------------------------------------------
     */

    /**
     * Update the state transition matrix F according to the new elapsed time.
     * Time is measured in seconds.
     * Update the process noise covariance matrix.
     */
    // compute the time elapsed between the current and previous measurements
    // dt - expressed in seconds
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
    // save the elapsed time
    previous_timestamp_ = measurement_pack.timestamp_;

    // create variables to use in formula
    double dt_2 = dt * dt;
    double dt_3 = dt_2 * dt;
    double dt_4 = dt_3 * dt;

    // Modify the F matrix so that the time is integrated
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;
    
    // Set the process covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ <<  (dt_4 * noise_ax)/4.,   0.,                     (dt_3 * noise_ax)/2.,   0.,
                0.,                     (dt_4 * noise_ay)/4.,   0.,                     (dt_3 * noise_ay)/2.,
                (dt_3 * noise_ax)/2.,   0.,                     dt_2* noise_ax,         0.,
                0.,                     (dt_3 * noise_ay)/2.,   0.,                     dt_2* noise_ay;
  
    // Call the Kalman Filter predict() function
    ekf_.Predict();

    /**----------------------------------------------------
     * Update
     *-----------------------------------------------------
      */

    /**
     * - Use the sensor type to perform the update step.
     * - Update the state and covariance matrices.
     */
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Use Radar measurements
        // We need calculate Jacobian matrix
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        // use covariance for radar
        ekf_.R_ = R_radar_;
        // call update
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        // Use Laser measurements
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        // call update
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}


