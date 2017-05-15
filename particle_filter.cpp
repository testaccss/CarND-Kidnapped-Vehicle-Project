/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 50;

    random_device rd;
    default_random_engine gen(rd());
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    for (int i = 0; i < num_particles; i++) {

        struct Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.d;

        particles[i] = p;
        weights[i] = (p.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    for(int i=0; i< num_particles; i++){
        if(yaw_rate == 0){
            particles[i].x = particles[i].x + (velocity * delta_t) * cos(particles[i].theta);
            particles[i].y = particles[i].y + (velocity * delta_t) * sin(particles[i].theta);
        }else{
            particles[i].x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta = particles[i].theta + (yaw_rate * delta_t);
        }


        random_device rd;
        default_random_engine gen(rd());
        normal_distribution<double> pos_error_x(particles[i].x, std_pos[0]);
        normal_distribution<double> pos_error_y(particles[i].y, std_pos[1]);
        normal_distribution<double> pos_error_theta(particles[i].theta, std_pos[2]);

        particles[i].x = pos_error_x(gen);
        particles[i].y = pos_error_y(gen);
        particles[i].theta = pos_error_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    for(int obs=0; obs<observations.size(); obs++){
        double obs_x = observations[obs].x;
        double obs_y = observations[obs].y;

        double temp_delta_l = 0.0;
        bool temp_delta_l_initialized = false;

        for(int l=0; l<predicted.size(); l++){
            double delta_x = obs_x - predicted[l].x;
            double delta_y = obs_y - predicted[l].y;

            double delta_l = sqrt(pow(delta_x, 2.0) + pow(delta_y, 2.0));

            if((!temp_delta_l_initialized) || (temp_delta_l > delta_l)) {
                temp_delta_l = delta_l;
                temp_delta_l_initialized = true;
                observations[obs].id = l;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
    for(int i=0; i<num_particles; i++){
        double current_x = particles[i].x;
        double current_y = particles[i].y;
        double current_theta = particles[i].theta;

        vector<LandmarkObs> predicted_landmarks;
        for(int l=0; l<map_landmarks.landmark_list.size(); l++){
            int l_id = map_landmarks.landmark_list[l].id_i;
            double l_x = map_landmarks.landmark_list[l].x_f;
            double l_y = map_landmarks.landmark_list[l].y_f;

            double delta_x = l_x - current_x;
            double delta_y = l_y - current_y;

            double distance = sqrt(pow(delta_x, 2.0) + pow(delta_y, 2.0));
            if(distance<=sensor_range){
                l_x = delta_x * cos(current_theta) + delta_y * sin(current_theta);
                l_y = delta_y * cos(current_theta) - delta_x * sin(current_theta);
                LandmarkObs landmark_in_range = {l_id, l_x, l_y};
                predicted_landmarks.push_back(landmark_in_range);
            }
        }

        dataAssociation(predicted_landmarks, observations);

        double new_weight = 1.0;
        for(int obs=0; obs<observations.size(); obs++) {
            int l_id = observations[obs].id;
            double obs_x = observations[obs].x;
            double obs_y = observations[obs].y;

            double delta_x = obs_x - predicted_landmarks[l_id].x;
            double delta_y = obs_y - predicted_landmarks[l_id].y;

            double numerator = exp(- 0.5 * (pow(delta_x,2.0)*std_landmark[0] + pow(delta_y,2.0)*std_landmark[1] ));
            double denominator = sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
            new_weight = new_weight * numerator/denominator;
        }
        weights[i] = new_weight;
        particles[i].weight = new_weight;

    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> resampled_particles;

    random_device rd;
    default_random_engine gen(rd());

    discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> resampled_particles((unsigned long) num_particles);

    for (int i = 0; i < num_particles; i++) {
        resampled_particles[i] = particles[dist_particles(gen)];
    }

    particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
    ofstream dataFile;
    dataFile.open(filename, ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
