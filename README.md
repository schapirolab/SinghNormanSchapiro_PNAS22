
# SinghNormanSchapiro_PNAS22
Models corresponding to Singh, Norman &amp; Schapiro (2022). Article and additional information available at 10.1073/pnas.2123432119.

These models have been developed in [Emergent](www.github.com/emer/emergent). The simulation files are set up to execute protocols described in the article and they will automatically default to their sequences.

## How to run
1. Install Go. Any version starting 1.13 should work. See instructions on how to do this [here](https://go.dev/dl/).
2. Clone this repository to your computer. See instructons on how to do this [here](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/cloning-a-repository).  
3. ```cd``` into the repository and into a simulation folder and run the command ```go build  && <simulation name>``` (either 'simulation_1' or 'simulation_2'). 
When you first run this command, Go should automatically download all dependencies required to run the simulation onto your computer. 
Once the command runs through, a GUI window should open up where you can interact with the model.

## Protocols for simulations

### Simulation 1
Click the "Train" button on the top left. The following sequence will be executed:
 1. The model will learn the satellite task in its awake state. You should see blocks of training and testing occuring sequentially until the model hits a learning criterion of 0.66 (66% accuracy on the task). 
 2.  The model will now switch to sleep and will begin replaying the information it learned during the wake state. At various points due to the endogenous dynamics of the model, it will fall into periods of high stability which the model will reinforce by contrasting it with immediately following periods of lower stability. The stability measure is displayed at the bottom of the screen as "AvgLaySim", and periods of high/low stability are marked as the "Plus" and "Minus" phases. Weight changes  resulting from the contrasts between these phases allow the model to learn during sleep.
 3. After 30,000 cycles of sleep, the model will switch back to a wake state and will immediately run a test block to measure changes in performance due to learning during sleep.
 4. The model will then reinitialize and run steps 1-3 again.
  
### Simulation 2:
Click the "Train" button on the top left. The following sequence will be executed:
1. Only neocortical and input/output layers will be turned on and the model will learn Env 1 items. Upon reaching 100% accuracy, the model will continue to learn Env 1 items for an additional 30 epochs.
2. After Env 1 items have been learnt by the neocortical layer, the hippocampus will be turned on and the full model will learn Env 2 items to 100% accuracy.
3. The model will now switch to sleep and run ten 10,000 cycle blocks of sleep. The blocks will consist of five NREM and REM blocks, alternated. After each sleep block, the model will run a test epoch to measure performance on Env 1 and Env 2 items.
4. The model will then reinitialize and run step 1-3 again.

## Editing model behavior
### Model outputs
The default behavior is to not produce output files, but outputs can be switched on by turning on output flags in `New()`.

Simulation 1 output flags:

`SlpWrtOut`: Write out all sleep cycle activities for all layers.

`TstWrtOut`: Write out all test epoch activities for all layers.

Simulation 2 output flags:

`SlpPatMatchWrtOut`: Write out decoded replay activity for all sleep cycles.

`TstWrtOut`: Write out all test epoch activities for all layers.

### Variables that control sleep behaviour:
The model relies on two mechanisms during sleep - (i) Short-term synaptic depression which destabilizes item attractors and (ii) Oscillating inhibition which reveals useful contrastive learning states in destabilized item attractors.

Synaptic depression is controlled by the "inc" and "dec" parameters which specify the rate of increase and recovery from synaptic depression over time, respectively. These can be edited in `SleepCycInit()`.

Layers in the network recieve either high or low amplitude oscillating inhibition. The amplitude for each is controlled via a sinusoidal equation which can be edited in `SleepTrial()` to change the various properties of the oscillations.


Please contact Dhairyya Singh (dsin@sas.upenn.edu) for additional questions.
