#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

/*--------------------------------------------------------------------------------------------------
 * Predict()
 * In this project, no need to use F Jacobian in prediction step, because we are using a
 * linear function in this step.
 *--------------------------------------------------------------------------------------------------
 */
void KalmanFilter::Predict() {
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

/*--------------------------------------------------------------------------------------------------
 * Update(...)
 * In this project, the measurement update for LIDAR will also use the regular Kalman filter
 * equations, since LIDAR uses linear equations.
 *--------------------------------------------------------------------------------------------------
 */
void KalmanFilter::Update(const VectorXd &z) {
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;  // Kalman gain

    // calculate new estimate
    x_ = x_ + (K * y);
    int x_size = x_.size();
    // create identity with properly shape
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

/*--------------------------------------------------------------------------------------------------
 * UpdateEKF(...)
 * In this project, the measurement update for RADAR sensor will use the EKF equations. So, we
 * need provide linearization using the Jacobian matrix equation, replacing H_ by Hj_
 *--------------------------------------------------------------------------------------------------
 */
void KalmanFilter::UpdateEKF(const VectorXd &z) {
    // we use the equations that map the predicted location z
    // from Cartesian coordinates to polar coordinates
    float px = x_[0];
    float py = x_[1];
    float vx = x_[2];
    float vy = x_[3];

    float ro = sqrt(px*px + py*py);
    float theta = atan2(py, px);
    float ro_dot;
    // check division by zero
    if(fabs(ro) < 0.0001) ro_dot = 0;
    else{
        ro_dot = (px*vx + py*vy)/ro;
        }
    VectorXd z_pred(3);
    z_pred << ro, theta, ro_dot;
    // update state
    VectorXd y = z - z_pred;
    // check -pi <= theta < pi
    y[1] = tools.CheckOrientation(y[1]);
    MatrixXd Ht = H_.transpose();
    // we are using Hj    
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;  // Kalman gain

    // calculate new estimate
    x_ = x_ + (K * y);
    int x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
