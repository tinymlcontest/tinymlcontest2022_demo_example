# Getting Start with STM32CubeAI

  ### Environment

  1. STM32CubeMX

     https://www.st.com/en/development-tools/stm32cubemx.html

     Availability as standalone software running on Windows®, Linux® and macOS® operating systems and 64-bit Java Runtime environment.

  2. MDK5

  3. ST-Link driver

     https://www.st.com/content/my_st_com/en/products/development-tools/software-development-tools/stm32-software-development-tools/stm32-utilities/stsw-link009.license=1656325086116.product=STSW-LINK009.version=2.0.2.html

  4. NUCLEO-L432KC

     Arm Cortex-M4 core at 80 MHz

     256 Kbytes of Flash memory, 64 Kbytes of SRAM

     Embedded ST-Link/V2-1 debugger/programmer

     ![L432](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/L432.jpg)

  

  ### New Project

  1. Open STM32CubeMX and click ***ACCESS TO BOARD SELECTER*** and select our board.

     ![mainmenu](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/mainmenu.png)

     ![select board](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/select%20board.png)

  2. Click ***Software Packs --- Manage Software Packs*** and 

     ![manage software packs](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/manage%20software%20packs.png)

     Click the check box of <u>Artifacial Intelligence( v7.1.0 )</u> under **X.CUBE.AI** and click ***Install Now.***

     ![install](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/install.png)

  3. Click ***Software Packs --- Select Components*** and set option as follow.

     ![components](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/components.png)

     ![Snipaste_2022-07-09_16-19-42](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/Snipaste_2022-07-09_16-19-42.png)

  4. Set PA2 as VCP_TX, and set PA15 as VCP_RX.

     ![pins](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/pins.png)

  5. Set Connectivity.

     ![connectivity](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/connectivity.png)

  6. Set AI network

     First, set the communication options as follow

     ![plantform setting](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/plantform%20setting.png)

     Then click ***Add network*** and import ours model by follow steps shown in the picture.

     ![image-20220627201730152](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220627201730152.png)

     After choosing model, you can click ***Analyze*** to view the resources needed to run the model.

     ![image-20220627202020441](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220627202020441.png)

  7. Set Clock

     ![clock](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/clock.png)

  8. Finish settings and click ***GENERATE CODE***

     ![project manager](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/project%20manager.png)

  ## Load Program to Board

  1. Connet the boadr to computer.

     <img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220627203515997.png" alt="image-20220627203515997" style="zoom: 25%;" />

  2. Open project in MDK5 and build.

     ![build](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/build.png)

  3. Check if the debugger is connected.

     First, click ***Options for Target***.

     ![targets](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/targets.png)

     Then switch to <u>Debug</u> and click ***Settings***.

     <img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/debug.png" alt="debug"  />

     If the debugger is connected, you can see the IDCODE and the Device Name. 

     <img src="https://raw.githubusercontent.com/AugustZTR/picbed/master/img/swdio.png" alt="swdio"  />

     Finally, switch to <u>Flash Download</u> and check <u>Reset and Run</u>

     ![full chip](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/full%20chip.png)

  4. Now you can load program to the board.

     ![load](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/load.png)

  ## Validation

  1. Click the reset button on the bottom of the board.

     ![image-20220627203955477](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220627203955477.png)

  2. Open CUbeMX project that we built before and switch to <u>network</u>

     ![image-20220627204238965](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220627204238965.png)

  3. Click ***Validate on target*** and you can see how the model runs on the board.

     ![image-20220627204444260](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220627204444260.png)

     ![image-20220627204542939](https://raw.githubusercontent.com/AugustZTR/picbed/master/img/image-20220627204542939.png)

 
 ## Metrics
 
 Two metrics, **Used Flash** and **duration**, will be extracted as the metrics to report the final scoring. 