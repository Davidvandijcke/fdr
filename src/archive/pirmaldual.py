// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2015, September 7 - October 6
// ###
// ###
// ### Thomas Moellenhoff, Robert Maier, Caner Hazirbas
// ###
// ###
// ###
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
// ###
// ###

import aux
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os
import time
import sys
import argparse

def parameterToFile(filename,repeats,gray,level,tauu,taum,sigmap,sigmas,lbmda_,nu,w,h,nc,available,total,t,iter_):
    file = open(filename, "w")
    if file == None:
        print("ERROR: Could not open file!")
    else:
        file.write("image: %d x %d x %d\n" % (w, h, nc))
        file.write("repeats: %d\n" % repeats)
        file.write("gray: %d\n" % gray)
        file.write("level: %d\n" % level)
        file.write("tauu: %f\n" % tauu)
        file.write("taum: %f\n" % taum)
        file.write("sigmas: %f\n" % sigmas)
        file.write("lbmda: %f\n" % lbmda_)
        file.write("nu: %f\n" % nu)
        file.write("GPU Memory: %d - %d = %f GB\n" % (total, available, float(total-available)/pow(10,9)))
        file.write("time: %f s\n" % t)
        file.write("iterations: %d\n" % iter_)
    file.close()
    
    def parameterToConsole(filename,repeats,gray,level,tauu,taum,sigmap,sigmas,lmbda,nu,w,h,nc,available,total,t,iter):
        print( "image: "+str(w)+" x "+str(h)+" x "+str(nc))
        print("repeats: "+str(repeats))
        print("gray: "+str(gray))
        print("level: "+str(level))
        print("tauu: "+str(tauu))
        print("taum: "+str(taum))
        print("sigmas: "+str(sigmas))
        print("lbmda: "+str(lmbda))
        print("nu: "+str(nu))
        print( "GPU Memory: "+str(total)+" - "+str(available)+" = "+str((total-available)/pow(10,9))+" GB")
        print( "time: "+str(t)+" s")
        print( "iterations: "+str(iter))
    
def energy(u, un, size):
    nrj = 0.0
    for i in range(size):
        nrj += abs(u[i] - un[i])
    return nrj

def bound(x1, x2, lbmda, k, l, f):
    return 0.25 * (x1*x1 + x2*x2) - lbmda * pow(k / l - f, 2)

def interpolate(k, uk0, uk1, l):
    return (k + (0.5 - uk0) / (uk1 - uk0)) / l

def on_parabola(u1,u2,u3,x1,x2,x3,f,lbmda,k,j,l):
    y = x3 + lbmda * pow(k / l - f, 2)
    norm = math.sqrt(x1*x1+x2*x2)
    v = 0.0
    a = 2.0 * 0.25 * norm
    b = 2.0 / 3.0 * (1.0 - 2.0 * 0.25 * y)
    d = b < 0 ? (a - math.pow(math.sqrt(-b), 3)) * (a + math.pow(math.sqrt(-b), 3)) : a*a + b*b*b
    c = math.pow((a + math.sqrt(d)), 1.0/3.0)
    if d >= 0:
        v = c == 0 ? 0.0 : c - b / c
    else:
        v = 2.0 * math.sqrt(-b) * math.cos((1.0 / 3.0) * math.acos(a / (math.pow(math.sqrt(-b), 3))))
    
    if norm == 0:
        u1[j], u2[j] = 0.0, 0.0
    else:
        u1[j] = (v / (2.0 * 0.25)) * x1 / norm
        u2[j] =  (v / (2.0 * 0.25)) * x1 / norm

    u3[j] = bound(u1[j], u2[j], lbmda, k, l, f)
    
def init(u, ubar, p1, p2, p3, s1, s2, mu1, mu2, mubar1, mubar2, f, h, w, l, proj, nc):
    x = threadIdx.x + blockDim.x * blockIdx.x;
    y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (x < w and y < h):
        I = 0;
        J = 0;
        img = 0.0;

        for c in range(0, nc):
            img = f[x+y*w+c*h*w]; # image value
            for k in range(0, proj):
                I = x+y*w+k*h*w+c*h*w*l; # index for u, ubar, p1, p2, p3
                J = x+y*w+k*h*w+c*h*w*proj; # index for s1, s2, mu1, mu2, mubar1, mubar2
                if (k<l):
                    u[I] = img;
                    ubar[I] = img;
                    p1[I] = 0.0;
                    p2[I] = 0.0;
                    p3[I] = 0.0;
                s1[J] = 0.0;
                s2[J] = 0.0;
                mu1[J] = 0.0;
                mu2[J] = 0.0;
                mubar1[J] = 0.0;
                mubar2[J] = 0.0;
    
