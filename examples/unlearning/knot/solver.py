"""
Solve the optimization problem to obtain an optimal solution
that maximizes the total similarity score across the board

The similarity matrix is a R * P matrix, where R is the number of reviewers, and P is
the number of papers
"""

from cvxopt import matrix, solvers, sparse, spmatrix

# from mosek import iparam

# Suppress extensive logging
# solvers.options['MOSEK'] = {mosek.iparam.log: 0}


def solve(
    workload_max,
    workload_min,
    paper_nominal,
    tpc_count,
    paper_count,
    similarity_matrix,
):
    print(
        "Solving the optimization problem with %d clusters and %d clients."
        % (tpc_count, paper_count)
    )

    # building matrix c for the optimization objective
    c = matrix(list(matrix(similarity_matrix).trans() * -1))

    # Use the following for testing with small matrices, using lists to represent them
    # c = matrix(list(matrix(similarity_matrix) * -1))

    # building matrix G for inequality constraints

    print("Producing matrix G: workload max constraints...")
    row_indices = []
    column_indices = []
    for i in range(tpc_count):
        for j in range(paper_count):
            row_indices.append(i)
            column_indices.append(j + i * paper_count)

    w_max = spmatrix(1.0, row_indices, column_indices)

    print("Producing matrix G: workload min constraints...")
    row_indices = []
    column_indices = []
    for i in range(tpc_count):
        for j in range(paper_count):
            row_indices.append(i)
            column_indices.append(j + i * paper_count)

    w_min = spmatrix(-1.0, row_indices, column_indices)

    print("Producing matrix G: x >= 0 constraints...")
    greater_than_zero = spmatrix(
        -1.0, range(tpc_count * paper_count), range(tpc_count * paper_count)
    )

    print("Producing matrix G: x <= 1 constraints...")
    less_than_one = spmatrix(
        1.0, range(tpc_count * paper_count), range(tpc_count * paper_count)
    )

    G = sparse([[w_max, w_min, greater_than_zero, less_than_one]])

    # Building matrix h for the right hand side of inequality constraints
    print("Producing matrix h for the right hand side of inequality constraints...")

    h_workload_upper = [workload_max for x in range(tpc_count)]
    h_workload_lower = [-workload_min for x in range(tpc_count)]

    h_greater_than_zero = [0.0 for x in range(tpc_count * paper_count)]
    h_less_than_one = [1.0 for x in range(tpc_count * paper_count)]

    h = matrix(
        list(
            h_workload_upper + h_workload_lower + h_greater_than_zero + h_less_than_one
        )
    )

    # building matrix A for the equality constraints
    print("Producing matrix A: reviews per paper equality constraints...")
    reviews_per_paper = spmatrix(1.0, range(paper_count), range(paper_count))
    reviews_per_paper_list = [[reviews_per_paper] for x in range(tpc_count)]
    reviews = sparse(reviews_per_paper_list)

    print("Producing matrix A: Conflicts of Interest equality constraints...")

    A = sparse([[reviews]])

    # building matrix b for the right hand side of equality constraints
    print("Producing matrix b for the right hand side of equality constraints...")
    b_reviews_per_paper = [paper_nominal * 1.0 for x in range(paper_count)]

    b = matrix(list(b_reviews_per_paper))

    # Run the LP solver.
    print("Starting the LP solver.")
    sol = solvers.lp(c, G, h, A, b, solver="mosek")
    # sol = solvers.lp(c, G, h, A, b)

    # Convert the solutions to a two-dimensional array,
    # with TPC members as rows, and papers as columns.

    # For example:
    # An assignment array of [[0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]
    # implies that TPC 0 is assigned paper 2, TPC 1 is assigned paper 1 and 3,
    # and TPC 2 is assigned paper 0 and 4.

    assignment = [[0 for x in range(paper_count)] for x in range(tpc_count)]

    for tpc in range(tpc_count):
        for paper in range(paper_count):
            if abs(sol["x"][tpc * paper_count + paper] - 1.0) < 0.01:
                assignment[tpc][paper] = 1
            elif abs(sol["x"][tpc * paper_count + paper] - 0.0) < 0.01:
                assignment[tpc][paper] = 0
            elif abs(sol["x"][tpc * paper_count + paper] - 0.5) < 0.01:
                assignment[tpc][paper] = 1

                print(
                    "Alert: The result for cluster '%d' and client '%d' is 0.5. "
                    " Its value is forcefully set to: %f"
                    % (tpc, paper + 1, assignment[tpc][paper])
                )
            else:
                assignment[tpc][paper] = round(sol["x"][tpc * paper_count + paper])
                print(
                    "Alert: The result for cluster '%d' and client '%d' is not an integer. "
                    " Its value is forcefully set to: %f"
                    % (tpc, paper + 1, assignment[tpc][paper])
                )

    return assignment
