
#include "S_encode.h"

#define S_DRIVE_EN   7
#define S_DRIVE_LPWM 6
#define S_DRIVE_RPWM 5

float Kp = 500.0;  // Proportional gain
float Ki = 0.0;  // Integral gain
float Kd = 0.0;  // Derivative gain
float S_previous_error = 0.0;
float S_integral_error = 0.0;


void init_SDrive(){
  pinMode( S_DRIVE_EN, OUTPUT);
  pinMode( S_DRIVE_RPWM, OUTPUT);
  pinMode( S_DRIVE_LPWM, OUTPUT);

  digitalWrite( S_DRIVE_EN, LOW);
  digitalWrite( S_DRIVE_RPWM, LOW);
  digitalWrite( S_DRIVE_LPWM, LOW);
}


void Steering_turn( int angular_z){
  int PWM_angular_z = angular_z*255/100;
  
  if ( PWM_angular_z < 0 && left_limit() == 0){
    // turn left
    digitalWrite( S_DRIVE_EN, HIGH);
    analogWrite( S_DRIVE_RPWM, -PWM_angular_z);
    analogWrite( S_DRIVE_LPWM, 0);
  }
  else if ( PWM_angular_z > 0 && right_limit() == 0){
    // turn right
    digitalWrite( S_DRIVE_EN, HIGH);
    analogWrite( S_DRIVE_RPWM, 0);
    analogWrite( S_DRIVE_LPWM, PWM_angular_z);
  }
  else{
    digitalWrite( S_DRIVE_EN, LOW);
    digitalWrite( S_DRIVE_RPWM, LOW);
    digitalWrite( S_DRIVE_LPWM, LOW);
  }
  
}


void stop_Sturn(){
  digitalWrite( S_DRIVE_EN, LOW);
  digitalWrite( S_DRIVE_RPWM, LOW);
  digitalWrite( S_DRIVE_LPWM, LOW);
}



int steering_drive( int angular_z){
  float control_value = mapToRange( angular_z, -100, 100, -0.532, 0.532);

  float en_value  = S_angel_read();

  float error = control_value - en_value;
  S_integral_error = S_integral_error + error;
  float derivative = error - S_previous_error;
  float pid_output = (Kp * error) + (Ki * S_integral_error) + (Kd * derivative);

  if ( pid_output > 100){
    pid_output = 100;
  }
  else if (pid_output < -100){
    pid_output = -100;
  }

  Steering_turn( pid_output);

  return pid_output;
}
