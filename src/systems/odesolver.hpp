namespace systems{
namespace discretization {

	template<typename System>
	struct LeapFrog
	{
		using T = typename System::ValueT;
		using State = typename System::State;

		LeapFrog(const System& _system, T _dt) : m_system(_system), m_dt(_dt) {}

		State operator()(const State& _state) const
		{
			State next{};
			const auto a0 = m_system.rhs(_state);
			next.position = _state.position + _state.velocity * m_dt + static_cast<T>(0.5) * a0 * m_dt * m_dt;

			const auto a1 = m_system.rhs(next);
			next.velocity = _state.velocity + static_cast<T>(0.5) * (a0 + a1) * m_dt;

			return next;
		}

		T deltaTime() const { return m_dt; }
	private:
		const System& m_system;
		T m_dt;
	};

	template<typename System>
	struct ForwardEuler
	{
		using T = typename System::ValueT;
		using State = typename System::State;

		ForwardEuler(const System& _system, T _dt) : m_system(_system), m_dt(_dt) {}

		State operator()(const State& _state)
		{
			State next;

			const auto a0 = m_system.rhs(_state);
			next.position = _state.position + _state.velocity * m_dt;
			next.velocity = _state.velocity + a0 * m_dt;

			return next;
		}

		T deltaTime() const { return m_dt; }
	private:
		const System& m_system;
		T m_dt;
	};
}}