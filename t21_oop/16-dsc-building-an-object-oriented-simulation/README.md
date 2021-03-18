
# Building an Object-Oriented Simulation

## Introduction

In this lesson, you'll learn a bit more about best practices for running simulations in the real world using object-oriented programming!

## Objectives

You will be able to:

* Use inheritance to write nonredundant code
* Create methods that calculate statistics of the attributes of an object
* Create object-oriented data models that describe the real world with multiple classes and subclasses and interaction between classes


## Creating Stochastic Simulations

As a capstone for everything you've learned about object-oriented programming, you'll be creating a **_Herd Immunity Simulation_**.


<img src='images/herd_immunity.gif'>


[gif created by u/theotheredmund](https://www.reddit.com/r/dataisbeautiful/comments/5v72fw/how_herd_immunity_works_oc/)

This simulation is meant to model the effects that vaccinations have on the way a communicable disease spreads through a population. The simulation you're building depends on just a few statistics from the CDC (Center for Disease Control):

* `r0`, the average number of people a contagious person infects before they are no longer contagious (because they got better or they died)
* `mortality_rate`, the percentage chance a person infected with a disease will die from it 

The main workflow of this simulation is as follows:

1. Create a `Person()` class with the following attributes:
    * `alive`
    * `vaccinated`
    * `is_infected`
    * `has_been_infected`
    * `newly_infected`
    
2. Create a `Simulation()` class with the following attributes:
    * `population`
    * `virus_name`
    * `num_time_steps`
    * `r0`
    * `percent_pop_vaccinated`

3. Create methods for our `Simulation()` class that will cover each step of the simulation. 

In order for our simulation to work, you'll need to define some rules for it:

* Each infected person will "interact" with 100 random people from the `population`. If the person the infected individual interacts with is sick, vaccinated, or has had the disease before, nothing happens. However, if the person the infected individual interacts with is healthy, unvaccinated, and has not been infected yet, then that person becomes infected. 

* At the end of each round, the following things happen:
    * All currently infected people either get better from the disease or die, with the chance of death corresponding to the mortality rate of the disease 
    * All people that were newly infected during this round become the new infected for the next round   
    
The simulation continues for the set number of time steps.  Any time someone dies or gets infected, log it in a text file called `'simulation_logs.txt'`.  Once the simulation is over, write some code to quickly parse the text logs into data and visualize the results, so that you can run multiple simulations and answer questions like: 

* If vaccination rates for {disease x} dropped by 5%, how many more people become infected in an epidemic? How many more die?
* What does the spread of {disease x} through a population look like?

If this all seems a bit daunting, don't worry! You'll be provided with much more detail as you build this step-by-step during the lab. 

With that, go ahead and take a look at this cool simulation lab!

## Summary

In this lesson, you reviewed some best practices for simulating things in the real world using object-oriented programming!
