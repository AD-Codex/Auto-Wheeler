
#include <Arduino.h>
#include <ros.h>
#include "F_drive.h"
#include "S_drive.h"
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>



int ros_linear_x = 0;
int ros_linear_y = 0;
int ros_linear_z = 0;
int ros_angular_x = 0;
int ros_angular_y = 0;
int ros_angular_z = 0;

void onTwist(const geometry_msgs::Twist &msg);

ros::NodeHandle  nh;

std_msgs::String str_msg_write;
ros::Publisher chatter("/ewheeler/feedback", &str_msg_write);

ros::Subscriber<geometry_msgs::Twist> sub("/ewheeler/cmd_vel", &onTwist);




void setup() {

//  Serial.begin(115200);

  init_SEncode();
  init_SDrive();
  init_FDrive();

  nh.initNode();
  nh.advertise(chatter);
  nh.subscribe(sub);

}

void loop() {

  forward_drive( ros_linear_x);
//  Steering_turn( -1*ros_angular_z);
  int out = steering_drive( -1*ros_angular_z);

  int angel_read = S_angel_read() * 100;

  char read_data[40];
  sprintf(read_data, "map value:%d pid out:%d", angel_read, out);
  ros_log(read_data); 

  
  nh.spinOnce();


  delay(10);
}

void onTwist(const geometry_msgs::Twist &msg) {
  ros_linear_x = msg.linear.x;
  ros_linear_y = msg.linear.y;
  ros_linear_z = msg.linear.z;
  ros_angular_x = msg.angular.x;
  ros_angular_y = msg.angular.y;
  ros_angular_z = msg.angular.z;

//  char read_data[40];
//  sprintf(read_data, "linear_x:%d anguler_z:%d", ros_linear_x, ros_angular_z);
//  ros_log(read_data); 
  
}


void ros_log(char* msg) {
  str_msg_write.data = msg;
  chatter.publish( &str_msg_write);
}
