# Deep RL Quadcopter Controller

*Teach a Quadcopter How to Fly!*

The **quadricopter** or **quadrirotor helicopter** is becoming an increasingly popular type of aircraft for both personal and professional use. Its maneuverability is useful for various applications, from short-range deliveries to cinematography, from search and rescue acrobatics.

Most quadricopters have 4 motors to provide boost, although some other models with 6 or 8 motors are also informally included in this category. Multiple thrust points with a center of gravity in the center increase stability and enable a variety of flight behaviors.

But there is a price for all this-the high complexity of controlling such an aircraft makes it almost impossible to control the momentum of each engine manually. Thus, most commercial quadricopters try to simplify flight control by accepting a single magnitude of boost and pitch / roll / yaw controls, making their control much more intuitive and fun.

The next step in this evolution is to enable quadricopters to assume desired control behaviors, such as takeoff and landing, autonomously. You could design these controls using a classical approach (say, implementing PID controllers). Or it is possible to use reinforcement learning to create agents who can learn these behaviors on their own.

In this project, an agent was developed to pilot a quadcopter and then train it using a reinforcement learning algorithm!

This project is part of Udacity's deep learning nanodegree program, if you are interested in checking out the original project, see this [repository](https://github.com/udacity/RL-Quadcopter-2.git).

## Software Dependencies

Make sure the `opencv-python`, `numpy`, `pandas`, `ipykernel`, `tensorflow`, `matplotlib` and `jupyter notebook` are installed:

`conda install opencv-python numpy pandas ipykernel tensorflow matplotlib jupyter notebook`

To check the software dependencies used in this project, see the link below: 

[requirements](requirements.txt)

To install software dependencies:

`conda install requirements.txt`

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.

```
git clone https://github.com/Italo-Pereira-Guimaraes/Deep-Learning-Nanodegree.git
cd Project 5 - Teaching a Flying Quaricopter
```

2. Create and activate a new environment.

```
conda create -n quadcop python=3.6 matplotlib numpy pandas
source activate quadcop
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `quadcop` environment. 
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

4. Open the notebook.
```
jupyter notebook Quadcopter_Project.ipynb
```

5. Before running code, change the kernel to match the `quadcop` environment by using the drop-down menu (**Kernel > Change kernel > quadcop**). Then, follow the instructions in the notebook.

## Evaluation

The project was evaluated according to the following [rubric](https://review.udacity.com/#!/rubrics/1189/view)

## license
 
For more information see:

[license](LICENSE.txt)

