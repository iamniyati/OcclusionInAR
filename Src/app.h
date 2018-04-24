#pragma once

//some often used C standard library header files
#include <cstdlib>
#include <cstdio>
#include <cstring>

//some often used STL header files
#include <iostream>
#include <memory>
#include <vector>

#include <Kinect.h>

//size of the window
#define SCRWIDTH 512
#define SCRHEIGHT 424

//some useful typedefs for explicit type sizes
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;
typedef char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

//safe way of deleting a COM object
template<typename T>
void SafeRelease(T& ptr) { if(ptr) { ptr->Release(); ptr = nullptr; } }

class App
{
public:
  void Init();
  void Tick(float deltaTime);
  void Shutdown();

  void SetPixelBuffer(uint32* pixelBuffer) { m_pixelBuffer = pixelBuffer; }

  //safe way of plotting a pixel
  void Plot(int x, int y, uint32 color)
  {
    if(x < 0 || x >= SCRWIDTH || y < 0 || y >= SCRHEIGHT)
      return;
    m_pixelBuffer[x + y * SCRWIDTH] = color;
  }

private:

  //pointer to buffer that contains pixels that get pushed to the screen
  //size of this buffer is SCRWIDTH * SCRWIDTH * sizeof(uint32)
  uint32* m_pixelBuffer = nullptr;

  IKinectSensor* m_sensor = nullptr;
  IDepthFrameReader* m_depthFrameReader = nullptr;
  IColorFrameReader* m_colorFrameReader = nullptr;
  ICoordinateMapper* m_coordinateMapper = nullptr;

  uint32* m_colorBuffer = nullptr;

  ColorSpacePoint* m_colorSpacePoints = nullptr;
  uint16 *m_depthBuffer = nullptr;
  int m_depthWidth = 0, m_depthHeight = 0;
};
