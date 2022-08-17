```C
#include <Servo.h>


Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;


char flag = '5';

void setup() {
  
  servo1.attach(2);     //맴버함수인 attach : 핀 설정
  servo2.attach(3);     //맴버함수인 attach : 핀 설정
  servo3.attach(4);     //맴버함수인 attach : 핀 설정
  servo4.attach(5);     //맴버함수인 attach : 핀 설정
  servo5.attach(6);     //맴버함수인 attach : 핀 설정
  Serial.begin(9600);
  
}

void loop() {
  while(Serial.available()>0)
  {
    flag = Serial.read();
  }

  
  if(flag == '0')
  {
    Serial.println('0');
    servo1.write(-90);
    servo2.write(-90);
    servo3.write(-90);
    servo4.write(-90);
    servo5.write(-90);

  
    delay(1000);
    
    flag = '5';
    }

   else if(flag == '1')
   {
    Serial.println('1');
    servo1.write(-90);

    servo3.write(-90);
    servo4.write(-90);
    servo5.write(-90);

    
    delay(1000);
    
    flag = '5';
    }

   else if(flag == '2')
   {
    Serial.println('2');
    servo1.write(-90);

    servo4.write(-90);
    servo5.write(-90);

    
    delay(1000);
    
    flag = '5';
    }

   
   else if(flag == '3')
   {
    Serial.println('3');
    servo1.write(-90);


    servo5.write(-90);

   
    delay(1000);
    
    flag = '5';
    }

   else if(flag == '4')
   {
    Serial.println('4');
    servo1.write(-90);

    
    delay(1000);
    
    flag = '5';
    }
    
    else if(flag == '5')
    {
      Serial.println('5');
      servo1.write(90);
      servo2.write(90);
      servo3.write(90);
      servo4.write(90);
      servo5.write(90);
      
      //delay(1000);
      
    }
    
    //delay(1000);
    
}
'''
