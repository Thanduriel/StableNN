namespace systems{
namespace discretization {

	template<typename System, typename Method>
	class ODEIntegrator
	{
	public:
		using T = typename System::ValueT;
		using State = typename System::State;

		ODEIntegrator(const System& _system, T _dt) : m_system(_system), m_dt(_dt) {}

		State operator()(const State& _state) const
		{
			return m_method(m_system, _state, m_dt);
		}

		T deltaTime() const { return m_dt; }
	protected:
		Method m_method;
		const System& m_system;
		T m_dt;
	};

	// first order
	struct ForwardEuler
	{
		template<typename System, typename State, typename T>
		State operator()(const System& _system, const State& _state, T _dt) const
		{
			return _state + _dt * _system.rhs(_state);
		}
	};


	struct SymplecticEuler
	{
		template<typename System, typename State, typename T>
		State operator()(const System& _system, const State& _state, T _dt) const
		{
			const auto a0 = _system.rhs(_state)[1];
			State next;
			next.velocity = _state.velocity + a0 * _dt;
			next.position = _state.position + next.velocity * _dt;

			return next;
		}
	};

	// second order
	class LeapFrog
	{
	public:
		template<typename System, typename State, typename T>
		State operator()(const System& _system, const State& _state, T _dt) const
		{
			State next{};
			const auto a0 = _system.rhs(_state)[1];
			next.position = _state.position + _state.velocity * _dt + static_cast<T>(0.5) * a0 * _dt * _dt;

			const auto a1 = _system.rhs(next)[1];
			next.velocity = _state.velocity + static_cast<T>(0.5) * (a0 + a1) * _dt;

			return next;
		}
	};
}}