**<arduino code>**
```C
#include <Servo.h>


int red1 = 2;
int yellow = 3;
int green = 4;
int blue = 5;
int white = 6;
int red2 = 8;

char flag = '6';

void setup() {
  

  Serial.begin(9600);
  
  pinMode(red1,OUTPUT);
  pinMode(yellow,OUTPUT);
  pinMode(green,OUTPUT);
  pinMode(blue,OUTPUT);
  pinMode(white,OUTPUT);
  pinMode(red2, OUTPUT);
}

void loop() {
  while(Serial.available()>0)
  {
    flag = Serial.read();
  }

  
  if(flag == '0')
  {
    Serial.println('0');


    digitalWrite(red1,HIGH);
    delay(1000);
    digitalWrite(red1,LOW);
    
    }

   else if(flag == '1')
   {
    Serial.println('1');


    digitalWrite(yellow,HIGH);
    delay(1000);
    digitalWrite(yellow,LOW);
    }

   else if(flag == '2')
   {
    Serial.println('2');


    digitalWrite(green,HIGH);
    delay(1000);
    digitalWrite(green,LOW);
    }

   
   else if(flag == '3')
   {
    Serial.println('3');


    digitalWrite(blue,HIGH);
    delay(1000);
    digitalWrite(blue,LOW);
    }

   else if(flag == '4')
   {
    Serial.println('4');

    digitalWrite(white,HIGH);
    delay(1000);
    digitalWrite(white,LOW);

    }
    
    else if(flag == '5')
    {
      Serial.println('5');

      digitalWrite(red2,HIGH);
      delay(1000);
      digitalWrite(red2,LOW);
    }
    
    delay(1000);
    
}  
  
```
