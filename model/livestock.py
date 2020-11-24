import model.disease as D
import numpy as np
import numpy.random as rand
from enum import Enum

# {{{ Gender enumeration
class Gender(Enum):
    """
    Gender for animals.
    """
    FEMALE = 1
    MALE = 2
# }}}

# {{{ Animal
class Animal:
    # {{{ constructor 
    def __init__(self, gender, herd, params):
        """
        Create a new animal with the given gender, a herd object that
        contains the set of animals it co-exists with, and the model
        parameters.
        """
        self.age = 0
        self.gender = gender
        self.params = params
        self.herd = herd
        self.params = params
        
        # breeding state variables
        self.fertile = False
        self.last_bred_age = None
        self.pregnant = False
        self.nursing = False

        # health state variables
        self.health = params['livestock']['initial_health']

        # disease state variables
        self.diseases = {}
        self.immunization_dates = {}
    # }}}
    
    # {{{ set_disease_state 
    def set_disease_state(self, disease, state, time):
        self.diseases[disease] = state
        if state == D.SIRV.V:
            self.immunization_dates[disease] = time.current_time
        else:
            self.immunization_dates[disease] = None
    # }}}

    # {{{ immunize 
    def immunize(self, disease, time):
        # if in the S or R state for the disease, transition to the V state.
        # if in the I state, too late.
        if self.diseases[disease] == D.SIRV.S or self.diseases[disease] == D.SIRV.R:
            self.diseases[disease] = D.SIRV.V
            self.immunization_dates[disease] = time.get_current_time()
    # }}}

    # {{{ step
    def step(self, time):
        """
        Timestep : age animal, and if animal is pregnant and passes the
        gestation period, give birth to a new animal for the herd.
        """
        self.age += time.step_size_weeks()

        # update disease state for vaccines wearing off
        for disease in self.diseases:
            if (disease in self.immunization_dates and 
                self.immunization_dates[disease] is not None):
                if (self.params['disease'][disease]['wearoff'] < time.weeks_since(self.immunization_dates[disease]) and
                    self.diseases[disease] == D.SIRV.V):
                    self.set_disease_state(disease, D.SIRV.S, time)

        # pregnant -> birth transition
        if (self.pregnant and
            self.age-self.last_bred_age >= self.params['livestock']['gestation_period']):
            # spawn child and add to herd
            child = self.spawn(time)
            self.herd.add(child)

            # toggle flags indicating pregnant
            self.pregnant = False
            self.nursing = True

        # nursing transition
        if (self.nursing and
            self.age-self.last_bred_age >= self.params['livestock']['gestation_period']+
                                           self.params['livestock']['nursing_period']):
            self.nursing = False

        # transition to fertile
        if (self.pregnant == False and
            self.nursing == False and
            self.gender == Gender.FEMALE and 
            self.age > self.params['livestock']['maturity'] and 
            self.fertile == False):
            self.fertile = True
    # }}}

    # {{{ can_breed
    def can_breed(self):
        return (self.gender == Gender.FEMALE and 
                self.fertile)
    # }}}
    
    # {{{ breed
    def breed(self):
        """ 
        Animal becomes pregnant. 
        """
        self.last_bred_age = self.age
        self.pregnant = True
        self.fertile = False
    # }}}

    # {{{ spawn
    def spawn(self, time):
        self.pregnant = False
        self.fertile = True

        # randomly pick gender of offspring
        if rand.rand() > self.params['livestock']['bull_probability']:
            gender = Gender.FEMALE
        else:
            gender = Gender.MALE

        child = Animal(gender, self.herd, self.params)
        # child starts susceptable to all diseases known
        for disease in self.diseases:
            child.set_disease_state(disease, D.SIRV.S, time)
        return child
    # }}}
# }}}

# {{{ Herd
class Herd:
    """
    A herd is a collection of animals.
    """
    # {{{ constructor
    def __init__(self, params):
        """ Animals. """
        self.animals = []

        self.params = params
    # }}}

    # {{{ step
    def step(self, tracker, time):
        # step 1: iterate through the set of animals to identify the
        #         number of males and breedable females
        males = []
        b_females = []
        for animal in self.animals:
            if animal.gender == Gender.MALE:
                males.append(animal)
            else:
                if animal.can_breed():
                    b_females.append(animal)

        # step 2: breed animals
        weekly_trials = time.weekly_probability()

        # TODO: what is probability of a single female getting pregnant?
        if len(males)+len(b_females) > 0:
            p_breed = float(len(males))/float(len(males)+len(b_females))
  
            day_of_year = time.day_of_year()
            p_date_scale = np.exp((-(day_of_year - self.params['livestock']['breed_date_mu'])**2)/(2*(self.params['livestock']['breed_date_sigma']**2)))
            #p_date_scale = 1.0
            #print(f'{day_of_year} => {p_date_scale}')
            for b_female in b_females:
                for _ in range(weekly_trials):
                    if b_female.health >= self.params['livestock']['min_breed_health'] and rand.rand() < p_date_scale*p_breed*self.params['livestock']['breed_pscale']:
                        b_female.breed()
                        break

        # step 3: step animals
        for animal in self.animals:
            animal.step(time)

        # step 4: cull herd due to old age or bad health
        cull_set = []
        for animal in self.animals:
            if animal.health <= 0.0:
                tracker.health_death()
                cull_set.append(animal)
            else:
                for _ in range(weekly_trials):
                    r = self.params['livestock']['death_sigma'] * rand.randn() + self.params['livestock']['death_mu']
                    if animal.age > r:
                        cull_set.append(animal)
                        tracker.oldage_death()
                        break

        for animal in cull_set:
            self.remove(animal)
    # }}}

    # {{{ population management

    # {{{ size 
    def size(self):
        return len(self.animals)
    # }}}

    # {{{ count_gender 
    def count_gender(self, gender):
        i = 0
        for animal in self.animals:
            if animal.gender == gender:
                i = i + 1
        return i
    # }}}

    # {{{ add
    def add(self, animal):
        """ Add a new animal. """
        self.animals.append(animal)
    # }}}

    # {{{ remove
    def remove(self, animal):
        """ Remove an animal from the herd. """
        self.animals.remove(animal)
    # }}}

    # {{{ immunize
    def immunize(self, d, time):
        for animal in self.animals:
            animal.set_disease_state(d, D.SIRV.V, time)
    # }}}

    # }}}

    # {{{ feeding
    # {{{ food_need
    def food_need(self, time):
        """ Calculate the amount of food needed for the
            collection of animals over the given time period. 
        """
        return self.size() * self.params['livestock']['eat'] * time.step_size_weeks()
    # }}}

    # {{{ feed
    def feed(self, units, time):
        """ Update the health of the herd based on consuming a set
            number of food units over some time period. """
        # no cows, so return
        if self.size() < 1:
            return

        # compute ratio of food available vs food needed
        rat = min(1.0, units / (self.food_need(time)))

        h_inc = self.params['livestock']['health_fed'] * rat * time.step_size_weeks()
        h_dec = self.params['livestock']['health_starve'] * (1.0 - rat) * time.step_size_weeks()

        for animal in self.animals:
            animal.health = animal.health + h_inc - h_dec
            # clamp health to range [0,1]
            animal.health = max(0.0, min(animal.health, 1.0))
    # }}}
    # }}}
# }}}
