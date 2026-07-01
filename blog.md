<div> <h1>Matrix Multiplication: An execution analysis</h1></div>


I have implemented a General Matrix MUltiplication(GEMM) on my sandy Bridge i3 2n gen old processor which achived execution time of less than 0.161s sec from 58.8 sec in naive, ALso while benchmakrking it with OPENBLAS an industry standard matrix multiplication library, my code achived the efficiey of more than 70.55% relative to the openBLAS whose executation time was approx 0.113 sec, Just 0.048sec more than mine.
It sounds like a rocket science at the first glance that how someone could achive that much efficiency close to OpenBLAS while performing General Matrix MUltiplication(GEMM), But trust me it's easier than it sounds, in this blog i'll walk you through each step of how i figured out to achive that much speed clso to OPENBLASS from first principal even if you don't know anyrthing related to it. However remember our main goal is not  to outperform OpenBLAS or achieve theoretical peak GFLOPS (explained below), but to analyze how execution behavior changes under different optimization techniques, algorithmic choices, and hardware constraints. So that we just don't really understand how to otpimise matrix but also understand how the hardware of our computer behaves and that's the most execiting part of this journey. But before we begin, get comfortable with pre-requisete and the terms given below, that would be used frequenlty and make it easier for you to understand


## Pre-requisite
Have a medium  Knwoldege of C and pointers
