import ddf.minim.analysis.*;
import ddf.minim.*;


Minim       minim;
AudioPlayer jingle;
FFT         fft;

void setup()
{
  size(1000, 500, P3D);
  
  minim = new Minim(this);
  
  // specify that we want the audio buffers of the AudioPlayer
  // to be 1024 samples long because our FFT needs to have 
  // a power-of-two buffer size and this is a good size.
  jingle = minim.loadFile("baige.mp3", 1024);
  
  // loop the file indefinitely
  jingle.loop();
  
  // create an FFT object that has a time-domain buffer 
  // the same size as jingle's sample buffer
  // note that this needs to be a power of two 
  // and that it means the size of the spectrum will be half as large.
  fft = new FFT( jingle.bufferSize(), jingle.sampleRate() );
  colorMode(RGB,255,255,255);
  
  
 
}



void draw()
{
  background(0);
  

  
  //stroke(255);
  int[][][] colors = {{{150, 116, 102}, {191, 127, 105}, {173, 113, 83}, {162, 131, 127}, {83, 84, 97}}, {{163, 139, 136}, {172, 130, 107}, {162, 118, 100}, {183, 133, 120}, {168, 128, 135}}, {{150, 84, 87}, {148, 106, 88}, {104, 93, 92}, {131, 104, 107}, {186, 163, 162}}, {{110, 105, 101}, {194, 141, 128}, {168, 141, 128}, {174, 135, 105}, {163, 87, 88}}, {{197, 140, 127}, {186, 129, 119}, {179, 135, 130}, {162, 154, 133}, {151, 109, 98}}, {{189, 132, 111}, {191, 113, 103}, {165, 104, 87}, {157, 127, 123}, {176, 150, 153}}, {{161, 123, 118}, {188, 150, 142}, {115, 87, 87}, {104, 110, 97}, {176, 161, 162}}, {{166, 157, 148}, {179, 168, 158}, {177, 168, 171}, {190, 129, 124}, {151, 109, 102}}, {{183, 110, 98}, {159, 132, 127}, {184, 127, 116}, {158, 97, 83}, {153, 106, 75}}, {{178, 160, 150}, {123, 93, 104}, {160, 121, 117}, {185, 116, 99}, {118, 97, 97}}, {{147, 128, 115}, {117, 103, 105}, {153, 134, 116}, {152, 119, 123}, {171, 135, 134}}, {{175, 160, 169}, {161, 147, 162}, {170, 158, 164}, {182, 125, 102}, {193, 142, 114}}, {{175, 117, 80}, {109, 104, 110}, {172, 146, 136}, {182, 133, 132}, {207, 168, 146}}, {{163, 137, 138}, {163, 91, 87}, {171, 140, 124}, {176, 133, 117}, {181, 149, 142}}, {{176, 107, 98}, {156, 124, 112}, {150, 124, 112}, {171, 127, 119}, {144, 121, 115}}, {{150, 144, 131}, {193, 131, 99}, {193, 158, 152}, {144, 90, 94}, {152, 90, 75}}, {{187, 102, 101}, {136, 123, 126}, {158, 121, 90}, {195, 139, 138}, {167, 141, 141}}, {{167, 161, 151}, {151, 120, 121}, {186, 153, 144}, {93, 122, 129}, {180, 144, 138}}, {{120, 103, 119}, {177, 109, 90}, {170, 124, 107}, {192, 141, 125}, {164, 94, 81}}};
  //int t=int(millis()/ 10000.0);
  int[][] nocolor = {{133,200,242},{94,186,242},{196,221,242},{225,255,255},{173,216,230}};
  int time=millis();
  int time_p = 5000;
  
  // perform a forward FFT on the samples in jingle's mix buffer,
  // which contains the mix of both the left and right channels of the file
  fft.forward( jingle.mix );
  
  for(int i = 0; i < fft.specSize(); i++)
  {
    // draw the line for frequency band i, scaling it up a bit so we can see it
    //line( i, height, i, height - fft.getBand(i)*8 );
    //line(x1,y1,x2,y2);
    noStroke();
    
    
    //line(0,height,width/2,500-fft.getBand(i)*8);
    //line(width,height,width/2,500-fft.getBand(i)*8);
    
    
    int r=int(random(5));
    //fill(colors[t][r][0],colors[t][r][1],colors[t][r][2]);
    
    //dongguan lyrics
    //if (time < 14970 +time_p){
    //  fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    //} else if (time >= 14970 +time_p && time < 16970 +time_p){
    //  fill(colors[0][r][0],colors[0][r][1],colors[0][r][2]); 
    //} else if (time >= 16970 +time_p && time < 17660 +time_p){
    //  fill(colors[1][r][0],colors[1][r][1],colors[1][r][2]); 
    //} else if (time >= 17660 +time_p && time < 22370 +time_p){
    //  fill(colors[2][r][0],colors[2][r][1],colors[2][r][2]); 
    //} else if (time >= 22370 +time_p && time < 26260 +time_p){
    //  fill(colors[3][r][0],colors[3][r][1],colors[3][r][2]); 
    //} else if (time >= 26260 +time_p && time < 28110 +time_p){
    //  fill(colors[4][r][0],colors[4][r][1],colors[4][r][2]); 
    //} else if (time >= 28110 +time_p && time < 31500 +time_p){
    //  fill(colors[5][r][0],colors[5][r][1],colors[5][r][2]); 
    //} else if (time >= 31500 +time_p && time < 45160 +time_p){
    //  fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    //} else if (time >= 45160 +time_p && time < 46790 +time_p){
    //  fill(colors[6][r][0],colors[6][r][1],colors[6][r][2]); 
    //} else if (time >= 46790 +time_p && time < 48760 +time_p){
    //  fill(colors[7][r][0],colors[7][r][1],colors[7][r][2]); 
    //} else if (time >= 48760 +time_p && time < 52550 +time_p){
    //  fill(colors[8][r][0],colors[8][r][1],colors[8][r][2]); 
    //} else if (time >= 52550 +time_p && time < 46390 +time_p){
    //  fill(colors[9][r][0],colors[9][r][1],colors[9][r][2]); 
    //} else if (time >= 46390 +time_p && time < 58300 +time_p){
    //  fill(colors[10][r][0],colors[10][r][1],colors[10][r][2]); 
    //} else if (time >= 58300 +time_p && time < 61300 +time_p){
    //  fill(colors[11][r][0],colors[11][r][1],colors[11][r][2]); 
    //} else if (time >= 61300 +time_p && time < 75590 +time_p){
    //  fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    //} else if (time >= 75590 +time_p && time < 79120 +time_p){
    //  fill(colors[12][r][0],colors[12][r][1],colors[12][r][2]); 
    //} else if (time >=79120 +time_p && time < 82570 +time_p){
    //  fill(colors[13][r][0],colors[13][r][1],colors[13][r][2]); 
    //} else if (time >= 82570 +time_p && time < 85570 +time_p){
    //  fill(colors[14][r][0],colors[14][r][1],colors[14][r][2]); 
    //} else if (time >= 85570 +time_p && time < 89750 +time_p){
    //  fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    //} else if (time >= 89750 +time_p && time < 91570 +time_p){
    //  fill(colors[15][r][0],colors[15][r][1],colors[15][r][2]); 
    //} else if (time >= 91570 +time_p && time < 105580 +time_p){
    //  fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    //} else if (time >= 105580 +time_p && time < 107470 +time_p){
    //  fill(colors[16][r][0],colors[16][r][1],colors[16][r][2]); 
    //} else if (time >= 107470 +time_p && time < 109260 +time_p){
    //  fill(colors[17][r][0],colors[17][r][1],colors[17][r][2]); 
    //} else if (time >= 109260 +time_p && time < 112680 +time_p){
    //  fill(colors[18][r][0],colors[18][r][1],colors[18][r][2]); 
    //} else if (time >= 112680 +time_p && time < 116510 +time_p){
    //  fill(colors[17][r][0],colors[17][r][1],colors[17][r][2]); 
    //} else if (time >= 116510 +time_p && time < 118660 +time_p){
    //  fill(colors[17][r][0],colors[17][r][1],colors[17][r][2]); 
    //} else if (time >= 118660 +time_p && time < 121660 +time_p){
    //  fill(colors[17][r][0],colors[17][r][1],colors[17][r][2]); 
    //} else if (time >= 121660 +time_p){
    //  fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    //} 
    
    //baige
    if (time < 145920 +time_p){
      fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    } else if (time >= 145920 +time_p && time < 149700 +time_p){
      fill(colors[0][r][0],colors[0][r][1],colors[0][r][2]); 
    } else if (time >= 149700 +time_p && time < 152940 +time_p){
      fill(colors[1][r][0],colors[1][r][1],colors[1][r][2]); 
    } else if (time >= 152940 +time_p && time < 156220 +time_p){
      fill(colors[2][r][0],colors[2][r][1],colors[2][r][2]); 
    } else if (time >= 156220 +time_p && time < 159570 +time_p){
      fill(colors[3][r][0],colors[3][r][1],colors[3][r][2]); 
    } else if (time >= 159570 +time_p && time < 172850 +time_p){
      fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    } else if (time >= 172850 +time_p && time < 176210 +time_p){
      fill(colors[4][r][0],colors[4][r][1],colors[4][r][2]); 
    } else if (time >= 176210 +time_p && time < 179550 +time_p){
      fill(colors[5][r][0],colors[5][r][1],colors[5][r][2]); 
    } else if (time >= 179550 +time_p && time < 182780 +time_p){
      fill(colors[6][r][0],colors[6][r][1],colors[6][r][2]); 
    } else if (time >= 182780 +time_p && time < 186160 +time_p){
      fill(colors[7][r][0],colors[7][r][1],colors[7][r][2]); 
    } else if (time >= 186160 +time_p && time < 199440 +time_p){
      fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    } else if (time >= 199440 +time_p && time < 202710 +time_p){
      fill(colors[8][r][0],colors[8][r][1],colors[8][r][2]); 
    } else if (time >= 202710 +time_p && time < 205940 +time_p){
      fill(colors[9][r][0],colors[9][r][1],colors[9][r][2]); 
    } else if (time >= 205940 +time_p && time < 209270 +time_p){
      fill(colors[10][r][0],colors[10][r][1],colors[10][r][2]); 
    } else if (time >= 209270 +time_p && time < 212660 +time_p){
      fill(colors[11][r][0],colors[11][r][1],colors[11][r][2]); 
    } else if (time >= 212660 +time_p && time < 225580 +time_p){
      fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    } else if (time >= 225580 +time_p && time < 229260 +time_p){
      fill(colors[12][r][0],colors[12][r][1],colors[12][r][2]); 
    } else if (time >= 229260 +time_p && time < 232490 +time_p){
      fill(colors[13][r][0],colors[13][r][1],colors[13][r][2]); 
    } else if (time >= 232490 +time_p && time < 235820 +time_p){
      fill(colors[14][r][0],colors[14][r][1],colors[14][r][2]); 
    } else if (time >= 235820 +time_p && time < 239260 +time_p){
      fill(colors[15][r][0],colors[15][r][1],colors[15][r][2]); 
    } else if (time >= 239260 +time_p){
      fill(nocolor[r][0],nocolor[r][1],nocolor[r][2]); 
    } 
    
    

    //stroke(colors[t][r][0],colors[t][r][1],colors[t][r][2]);
    //line(width/2-fft.getBand(i)*2,height/2-fft.getBand(i)*2,width/2+fft.getBand(i)*2,height/2-fft.getBand(i)*2);
    //line(width/2-fft.getBand(i)*2,height/2-fft.getBand(i)*2,width/2-fft.getBand(i)*2,height/2+fft.getBand(i)*2);
    //line(width/2-fft.getBand(i)*2,height/2+fft.getBand(i)*2,width/2+fft.getBand(i)*2,height/2+fft.getBand(i)*2);
    //line(width/2+fft.getBand(i)*2,height/2+fft.getBand(i)*2,width/2+fft.getBand(i)*2,height/2-fft.getBand(i)*2);
    
    ellipse(i*20,height/2,fft.getBand(i)*3,fft.getBand(i)*3);
    
    //int x=int(random(500));
    //int y=int(random(500));
    //ellipse(x,y,fft.getBand(i)*3,fft.getBand(i)*3);
  }
}
