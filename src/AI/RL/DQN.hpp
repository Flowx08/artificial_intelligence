#ifndef DQN_HPP
#define DQN_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "../deeplearning/NeuralNetwork.hpp"
#include "../deeplearning/OptimizerSDG.hpp"
#include "../util/Tensor.hpp"
#include "IntelligentAgent.hpp"
#include "Environment.hpp"
#include <vector> 

////////////////////////////////////////////////////////////
///	AI
////////////////////////////////////////////////////////////
namespace ai
{

	struct Experience
	{
		Tensor_float state;
		Tensor_float action;
		Tensor_float state_next;
		int actionid;
		float reward;
	};

	class DQN : public IntelligentAgent
	{
	public:
		////////////////////////////////////////////////////////////
		/// \brief	Create an intelligent agent from parameters	
		///
		////////////////////////////////////////////////////////////
		DQN(ai::NeuralNetwork& net, int short_term_mem_size, float learningrate, float curiosity, float longtermexploitation);
		
		////////////////////////////////////////////////////////////
		/// \brief	Receive the current world state and return the
		/// best action id
		///
		////////////////////////////////////////////////////////////
		int act(Tensor_float& state);
		
		////////////////////////////////////////////////////////////
		/// \brief	Receive the current world state and return the
		/// best action id
		///
		////////////////////////////////////////////////////////////
		int act(Tensor_float& state, const std::vector<bool>& allowed_actions);

		////////////////////////////////////////////////////////////
		/// \brief Teach the agent new things by manually surf
		/// the state-action space
		///
		////////////////////////////////////////////////////////////
		void teach(Tensor_float& state, int actionid);

		////////////////////////////////////////////////////////////
		/// \brief	Receive the new world state and the old action
		/// reward. Update the short term memory and the internal model
		/// of the world
		///
		////////////////////////////////////////////////////////////
		void observe(Tensor_float &newstate, float oldreward);
		
		////////////////////////////////////////////////////////////
		/// \brief	Set the curiosity of the agent, range: (1.0, 0.0).
		/// The higher it is the more randomly it will explore,
		/// the lower it is the more it will exploit what he
		/// know to maximize the reward
		///
		////////////////////////////////////////////////////////////
		void setcuriosity(double curiosity);
		
		////////////////////////////////////////////////////////////
		/// \brief	Set the long term exploitation factor of the
		/// agent, range: (1.0, 0.0). If it's a high value it will
		/// work also for future rewards else it will work only for
		/// immediate rewards
		///
		////////////////////////////////////////////////////////////
		void setlongtermexploitation(double longtermexploitation);
		
		////////////////////////////////////////////////////////////
		/// \brief	Set the learning rate of the agent	
		///
		////////////////////////////////////////////////////////////
		void setlearningrate(double learningrate);

		////////////////////////////////////////////////////////////
		/// \brief	Get the last action id performed by the agent	
		///
		////////////////////////////////////////////////////////////
		int getlastaction();

		////////////////////////////////////////////////////////////
		/// \brief	Get current experience informations	
		///
		////////////////////////////////////////////////////////////
		const Experience getcurrentexperience();

		////////////////////////////////////////////////////////////
		/// \brief	Get curiosity value of the agent	
		///
		////////////////////////////////////////////////////////////
		double getcuriosity();

		////////////////////////////////////////////////////////////
		/// \brief	Get long term exploitation factor value of the
		/// agent
		///
		////////////////////////////////////////////////////////////
		double getlongtermexploitation();
		
		////////////////////////////////////////////////////////////
		/// \brief	Get learning rate of the agent	
		///
		////////////////////////////////////////////////////////////
		double getlearningrate();

		////////////////////////////////////////////////////////////
		/// \brief	Get the neural net of the agent	
		///
		////////////////////////////////////////////////////////////
		const NeuralNetwork& getneuralnet();

	private:
		////////////////////////////////////////////////////////////
		/// \brief	Get position of the max value in a tensor
		///
		////////////////////////////////////////////////////////////
		int getmaxpos(Tensor_float &data);
		
		////////////////////////////////////////////////////////////
		/// \brief	Get position of the max value in a tensor with
		/// mask, only the elements with a mask value equal to 1
		/// are used
		///
		////////////////////////////////////////////////////////////
		int getmaxpos(Tensor_float& data, const std::vector<bool>& mask);

		ai::NeuralNetwork* _internalmodel;				//Internal model of the world
		ai::OptimizerSDG _optimizer;					//Model optimizer
		Experience _experience_current;					//Current experience informations
		int _actionid;									//id of the current performed action
		std::vector<Experience> _short_term_mem;		//short term memory of experiences
		int _short_term_mem_size;						//short term memory buffer size
		int _short_term_mem_pos;						//short term memory current position
		int _short_term_mem_sampling_size;				//size of memory sampling
		double _curiosity;								//curiosity factor
		double _learningrate;							//learning rate factor
		double _longtermexploitation;					//long term exploitation factor
	};

} //namespace AI

#endif /* end of include guard: DQN_HPP */

