-module(neuron).
-export([trainer/6, generate_weights/1, generate_traning_set/1, feedforward/2, square_trainer/3, line_trainer/3, test_square_perceptron/2, training_is_over_line_function/4, test_line_perceptron/2]).

% Generate a list of random start weights
generate_weights(Inputs) ->
	generate_weights([random:uniform()*plus_or_minus()], Inputs -1).
generate_weights(Weights, Inputs) when Inputs > 0-> 
	generate_weights(Weights ++ [random:uniform()*plus_or_minus()], Inputs - 1);	
generate_weights(Weights, _) ->
	Weights.


%
% Input a list of weights and values
%
feedforward([InputsH|InputsT],[WeightsH|WeightsT]) ->
	feedforward(InputsT, WeightsT, InputsH * WeightsH).
feedforward([InputsH|InputsT],[WeightsH|WeightsT], Sum) ->
	feedforward(InputsT, WeightsT, Sum + (InputsH * WeightsH));
feedforward([],[],Sum) -> 
	activate(Sum).

activate(Sum) when Sum >= 0 -> 1;
activate(_) -> -1.


% Randomize if generated float is positive or negative  
plus_or_minus() -> 
	plus_or_minus(random:uniform()).
plus_or_minus(Modifier) when Modifier > 0.5 -> 1;
plus_or_minus(_) -> -1.

training_square_function(X,Y,Width,TestX,TestY) when (TestX >= X) and (TestY >= Y) 
										and (X + Width >= TestX) and (Y + Width >= TestY) ->
	1;
training_square_function(_,_,_,_,_) ->
	-1.

training_is_over_line_function(Slope,Yintercept,TestX,TestY) when (TestY >= (TestX*Slope) + Yintercept )->
	-1;
training_is_over_line_function(_,_,_,_) ->
	1.

%
% Adjust the perceptrons weight according to error and input and learning constant
% 
train_perceptron([InputsH|InputsT], [WeightsH|WeightsT], Learning_constant, Error) ->
	%io:format("Setting neuron to ~f from error ~w and previous ~f. ", [WeightsH + (Learning_constant * Error) * InputsH, Error, WeightsH]),
	train_perceptron(InputsT, WeightsT, Learning_constant, Error, 
		[WeightsH + (Learning_constant * Error) * InputsH]).
train_perceptron([InputsH|[]], [WeightsH|[]], Learning_constant, Error, Adjusted_weights) ->
    %io:format("Weighting bias. ~f", [WeightsH - Learning_constant * Error]),
	train_perceptron([], [], Learning_constant, Error,
		Adjusted_weights ++ [WeightsH + Learning_constant * Error]);
train_perceptron([InputsH|InputsT], [WeightsH|WeightsT], Learning_constant, Error, Adjusted_weights) ->
	%io:format("Weighting neuron. "),
	train_perceptron(InputsT, WeightsT,Learning_constant, Error, 
		Adjusted_weights ++ [WeightsH + (Learning_constant * Error) * InputsH]);
train_perceptron([], [],_, _, Adjusted_weights) ->
	%io:format("Exiting. \n"),
	Adjusted_weights.

%line_trainer(Weights,0,_)->
%	Weights;
%line_trainer(Weights,Iterations,Learning_constant) ->
%	X = random:uniform(100),
%	Y = random:uniform(100),
%	Bias = 1,
%	io:format("Guess: ~w",[Guess = feedforward([X,Y,Bias],Weights)]),
%	io:format("Desired: ~w", [Desired = training_is_over_line_function(-1,100,X,Y)]),
%	Error = Desired - Guess,
%	Adjusted_weights = train_perceptron([X,Y,Bias], Weights, Learning_constant, Error),
	%
%	print_weights(Adjusted_weights),
	%test_line_perceptron(Weights,10),
%	line_trainer(Adjusted_weights, Iterations -1, Learning_constant).

% Must be supplied a function with the arguments (Weights, Trainingset and Learnign_constat)

trainer(Max_iterations, Converge_to, Weights, Training_set, Learning_constant, Trainer_function) when is_function(Trainer_function) ->
	trainer(Max_iterations, Converge_to, Weights, Training_set, Learning_constant, Trainer_function, {0, Weights, 0, Max_iterations}).


trainer(0, Converge_to, Weights, Training_set, Learning_constant, Trainer_function, {Global_error, Adjusted_weights, RMS, Max_iterations}) when is_function(Trainer_function) ->
{Global_error, Adjusted_weights, RMS, 0};

trainer(Max_iterations, Converge_to, Weights, Training_set, Learning_constant, Trainer_function, {_, _, _, _}) when is_function(Trainer_function) ->
	
	{Global_error, Adjusted_weights} = Trainer_function(Weights, Training_set, Learning_constant),
	RMS = math:sqrt(Global_error / length(Training_set)),
	% Check if the converging point is reached
	if 
		 RMS >= Converge_to ->
		 	io:format("Continuing.."),  
			trainer(Max_iterations-1, 
				Converge_to, 
				Adjusted_weights, 
				Training_set, 
				Learning_constant, 
				Trainer_function, 
				{Global_error, Adjusted_weights, RMS, Max_iterations});

		RMS < Converge_to -> 
			io:format("Quitting.."),
			{Global_error, Adjusted_weights, RMS, Max_iterations, Converge_to}
	end.


line_trainer(Weights, [{X,Y,Desired}|TST], Learning_constant)->
	line_trainer(Weights, [{X,Y,Desired}|TST], Learning_constant, 0).

line_trainer(Weights,[],_,Global_error) ->
	{Global_error, Weights};
line_trainer(Weights, [{X,Y,Desired}|TST], Learning_constant, Global_error)->
	Bias = 1,
	Error = Desired - feedforward([X,Y,Bias],Weights),
	Adjusted_weights = train_perceptron([X,Y,Bias], Weights, Learning_constant, Error),
	print_weights(Adjusted_weights),
	line_trainer(Adjusted_weights, TST, Learning_constant, Global_error + Error * Error).



test_line_perceptron(Weights, Iterations) -> 
	test_line_perceptron(Weights, Iterations, 0).
test_line_perceptron(_ , 0, Hits) ->
	io:format("Precision of: ~w\n", [Hits]),
	Hits;
test_line_perceptron(Weights, Iterations, Hits) -> 
	X = random:uniform(100),
	Y = random:uniform(100),
	Bias = 1,
	Guess = feedforward([X,Y,Bias],Weights),

	case Guess of
		 1 -> io:format("Guessed that ~w =X and ~w = Y is over the line. ", [X,Y]);
		-1 -> io:format("Guessed that ~w =X and ~w = Y is under the line. ",[X,Y])
	end,

	Desired = training_is_over_line_function(-1,100,X,Y),
	case Desired - Guess of
		0 -> io:format("Which was RIGHT\n"),
			 test_line_perceptron(Weights, Iterations -1, Hits + 1);
		_ -> io:format("Which was WRONG\n"),
			 test_line_perceptron(Weights, Iterations -1, Hits)
	end.

square_trainer(Weights,0,_) ->
	Weights;
square_trainer(Weights, Iterations, Learning_constant) -> 
	test_square_perceptron(Weights, 100),
	X = random:uniform(100),
	Y = random:uniform(100),
	Bias = 1,
	Guess = feedforward([X,Y,Bias],Weights),
	Desired = training_square_function(5,5,50,X,Y),
	Error = Desired - Guess,
	Adjusted_weights = train_perceptron([X,Y,Bias], Weights, Error, Learning_constant),
	%print_weights(Adjusted_weights),
	square_trainer(Adjusted_weights, Iterations -1, Learning_constant).

%
% Function to test the validity of the set of weights of the perceptron.
test_square_perceptron(Weights, Iterations) -> 
	test_square_perceptron(Weights, Iterations, 0).
test_square_perceptron(_ , 0, Hits) ->
	io:format("Precision of: ~w\n", [Hits]),
	Hits;

test_square_perceptron(Weights, Iterations, Hits) -> 
	X = random:uniform(100),
	Y = random:uniform(100),
	Bias = 1,
	Guess = feedforward([X,Y,Bias],Weights),

	case Guess of
		 1 -> io:format("~w =X and ~w = Y is within the box\n",[X,Y]);
		-1 -> io:format("~w =X and ~w = Y is NOT within the box\n",[X,Y])
	end,

	Desired = training_square_function(0,0,50,X,Y),
	case Desired - Guess of
		0 -> test_square_perceptron(Weights, Iterations -1, Hits + 1);
		_ -> test_square_perceptron(Weights, Iterations -1, Hits)
	end.

print_weights([])->
	io:format("\n");
print_weights([H|T]) ->
	io:format("~f   ", [H]),
	print_weights(T).


generate_traning_set(Size) ->
	X = random:uniform(100),
	Y = random:uniform(100),
	generate_traning_set(Size -1, [{X,Y, training_is_over_line_function(-1, 100, X,Y)}]).
generate_traning_set(0, Training_set)->
	Training_set;
generate_traning_set(Size, Training_set) ->
	X = random:uniform(100),
	Y = random:uniform(100),
	generate_traning_set(Size -1, Training_set ++ [{X,Y, training_is_over_line_function(-1, 100, X,Y)}]).