from sympy.utilities.iterables import multiset_permutations

def parse_string(input_string):
    stmp = input_string.split('/')
    fs = stmp[0].split('_G')[1]
    op = stmp[1].split('_Gamma')[1]
    mom = stmp[3]
    return fs, op, mom


def eps_over_k(fs, op, mom):
    cycl = ['X', 'Y', 'Z']
    pos = ['012', '120', '201']
    neg = ['021', '210', '102']
    iV = cycl.index(fs)
    iJ = cycl.index(op)
    
    # If the directions are the same, return 0 (antisymmetry)
    if iV == iJ:
        return 0.0
    
    # Determine the momentum component that is non-zero
    iMom = 3 - iV - iJ
    kComp = float(mom.split('_')[iMom])
    
    # If the component is 0, return 0
    if kComp == 0.0:
        return 0.0
    
    # Determine sign based on position (cyclindrical symmetry)
    epsStr = str(iV) + str(iJ) + str(iMom)
    if epsStr in pos:
        sign = 1.0
    elif epsStr in neg:
        sign = -1.0
    else:
        sign = 0.0
    
    return sign / kComp


def get_moms_and_prefacs_V():
    nSqs = [0, 1, 2, 3, 4, 5]
    cycl = ['X', 'Y', 'Z']
    momss = [
        [[0, 0, 0]],  # nSq=0
        [[1, 0, 0], [-1, 0, 0]],  # nSq=1
        [[1, 1, 0], [1, -1, 0], [-1, -1, 0]],  # nSq=2
        [[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [-1, -1, -1]],  # nSq=3
        [[2, 0, 0], [-2, 0, 0]],  # nSq=4
        [[2, 1, 0], [2, -1, 0], [-2, 1, 0], [-2, -1, 0]]  # nSq=5
    ]
    
    momentum_list = []
    prefactor_list = []
    
    # Generate momenta and prefactors
    for nSq in nSqs:
        moms = momss[nSq]
        
        # Skip nSq=0 because the momentum list should be empty
        if nSq == 0:
            momentum_list.append([])
            prefactor_list.append([])
            continue
        
        temp_mom = []
        temp_prefacs = []
        
        for gPol in cycl:
            iV = cycl.index(gPol)
            for gJ in cycl:
                iJ = cycl.index(gJ)
                iMom = 3 - iV - iJ  # This component must not be 0
                
                if gPol == gJ:
                    continue
                
                string = f"final_state_G{gPol}/operator_Gamma{gJ}/n2_{nSq}/"
                
                for m in moms:
                    for p in multiset_permutations(m):
                        if p[iMom] == 0:
                            continue
                        
                        # Create the final state string
                        strFinal = string + f"{p[0]}_{p[1]}_{p[2]}"
                        temp_mom.append(strFinal)
                        
                        # Calculate the prefactor for this momentum
                        a, b, c = parse_string(strFinal)
                        prefac = eps_over_k(a, b, c)
                        temp_prefacs.append(prefac)
        
        momentum_list.append(temp_mom)
        prefactor_list.append(temp_prefacs)
    
    return momentum_list, prefactor_list

def get_moms_and_prefacs_A0():
    directions = ["GX", "GY", "GZ"]
    prefactors = {"GX": 1, "GY": -1, "GZ": 1}
    momentum_list = []
    prefactor_list = []

    for nsq in range(6):
        nsq_momentum = []
        nsq_prefactors = []

        for dx in range(-nsq, nsq + 1):
            for dy in range(-nsq, nsq + 1):
                for dz in range(-nsq, nsq + 1):
                    if dx**2 + dy**2 + dz**2 == nsq:  # Check for correct n^2
                        for direction in directions:
                            if (direction == "GX" and dx != 0) or \
                               (direction == "GY" and dy != 0) or \
                               (direction == "GZ" and dz != 0):
                                element = (
                                    f"final_state_{direction}/operator_Gamma{direction[-1]}Gamma5/"
                                    f"n2_{nsq}/{dx}_{dy}_{dz}"
                                )
                                nsq_momentum.append(element)
                                nsq_prefactors.append(prefactors[direction])

        momentum_list.append(nsq_momentum)
        prefactor_list.append(nsq_prefactors)

    return momentum_list, prefactor_list

def get_moms_and_prefacs_A1():
    directions = ["GX", "GY", "GZ"]
    prefactors = {"GX": 1, "GY": -1, "GZ": 1}
    momentum_list = []
    prefactor_list = []

    for nsq in range(6):
        nsq_momentum = []
        nsq_prefactors = []

        # nsq=0
        if nsq == 0:
            element = "final_state_GX/operator_GammaXGamma5/n2_0/0_0_0"
            nsq_momentum.append(element)
            nsq_prefactors.append(prefactors["GX"])
            element = "final_state_GY/operator_GammaYGamma5/n2_0/0_0_0"
            nsq_momentum.append(element)
            nsq_prefactors.append(prefactors["GY"])
            element = "final_state_GZ/operator_GammaZGamma5/n2_0/0_0_0"
            nsq_momentum.append(element)
            nsq_prefactors.append(prefactors["GZ"])

        else:
            for dx in range(-nsq, nsq + 1):
                for dy in range(-nsq, nsq + 1):
                    for dz in range(-nsq, nsq + 1):
                        if dx**2 + dy**2 + dz**2 == nsq:  
                            # final state/operator component = 0
                            for direction in directions:
                                if direction == "GX" and dx != 0: continue
                                if direction == "GY" and dy != 0: continue
                                if direction == "GZ" and dz != 0: continue

                                element = (
                                    f"final_state_{direction}/operator_Gamma{direction[-1]}Gamma5/"
                                    f"n2_{nsq}/{dx}_{dy}_{dz}"
                                )
                                nsq_momentum.append(element)
                                nsq_prefactors.append(prefactors[direction])

        momentum_list.append(nsq_momentum)
        prefactor_list.append(nsq_prefactors)

    return momentum_list, prefactor_list

'''
def get_moms_and_prefacs_A2():
    mom1,pref1=get_moms_and_prefacs_A0()
    mom2,pref2=get_moms_and_prefacs_A1()
    return mom1,pref1,mom2,pref2
'''


def combine_momentum_lists():
    list_A,preA=get_moms_and_prefacs_A0()
    list_B,preB=get_moms_and_prefacs_A1()
    combined_lists = []
    
    # Function to extract momentum values (e.g., "1_0_0" from the full string)
    def get_momentum(path):
        parts = path.split('/')
        return parts[-1]
    
    # Function to extract nsq value
    def get_nsq(path):
        parts = path.split('/')
        for part in parts:
            if part.startswith('n2_'):
                return int(part[3:])
        return None
    
    # Find maximum nsq value to determine the size of our result list
    max_nsq = 0
    for sublist in list_A + list_B:
        for item in sublist:
            nsq = get_nsq(item)
            max_nsq = max(max_nsq, nsq)
    
    # Initialize the result list with empty lists for each nsq value
    combined_lists = [[] for _ in range(max_nsq + 1)]
    
    # Group elements by nsq and momentum
    for nsq in range(max_nsq + 1):
        # Get all elements with this nsq value from both lists
        nsq_elements_A = []
        nsq_elements_B = []
        
        for sublist in list_A:
            for item in sublist:
                if get_nsq(item) == nsq:
                    nsq_elements_A.append(item)
        
        for sublist in list_B:
            for item in sublist:
                if get_nsq(item) == nsq:
                    nsq_elements_B.append(item)
        
        # Create momentum dictionaries for this nsq
        momentum_dict_A = {}
        momentum_dict_B = {}
        
        # Group elements by momentum
        for item in nsq_elements_A:
            momentum = get_momentum(item)
            if momentum not in momentum_dict_A:
                momentum_dict_A[momentum] = []
            momentum_dict_A[momentum].append(item)
        
        for item in nsq_elements_B:
            momentum = get_momentum(item)
            if momentum not in momentum_dict_B:
                momentum_dict_B[momentum] = []
            momentum_dict_B[momentum].append(item)
        
        # Create all possible combinations for matching momenta
        for momentum in set(momentum_dict_A.keys()) & set(momentum_dict_B.keys()):
            for item_A in momentum_dict_A[momentum]:
                for item_B in momentum_dict_B[momentum]:
                    combined_lists[nsq].append([item_A, item_B])
    
    return combined_lists

def compute_combined_prefactors(x):
    combined_lists=combine_momentum_lists()
    def compute_prefactor_A0(entry):
        # Map for final state directions
        direction_map = {"GX": 1, "GY": 2, "GZ": 3}
        # Map for operator directions
        operator_map = {"GammaXGamma5": 1, "GammaYGamma5": 2, "GammaZGamma5": 3}
        
        # Split the string into its components
        parts = entry.split("/")
        final_state = parts[0]  # e.g., final_state_GX
        operator = parts[1]    # e.g., operator_GammaXGamma5
        momentum_components = parts[3]  # e.g., "1_0_0"
        
        # Extract the direction from the final_state
        direction = final_state.split("_")[2]  # Extract GX, GY, GZ
        # Extract the operator direction
        operator_direction = operator.split("_")[1]  # Extract GammaXGamma5, etc.
        
        # Get l and j from the direction and operator_direction
        l = direction_map[direction]
        j = operator_map[operator_direction]
        
        # Extract momentum components dx, dy, dz
        dx, dy, dz = map(int, momentum_components.split("_"))
        momentum_values = [dx, dy, dz]
        
        # Calculate prefactor: jth * lth / x and include -1 if j == l
        prefactor = (momentum_values[j - 1] * momentum_values[l - 1]) / x
        if j == l:
            prefactor -= 1
            
        return prefactor

    def get_prefactors(entry_pair):
        first_entry = entry_pair[0]
        second_entry = entry_pair[1]
        
        # First prefactor: based on X/Z vs Y in first string
        has_Y_first = "final_state_GY" in first_entry
        prefactor1 = -1 if has_Y_first else 1
        
        # Second prefactor: computed from function on first string
        prefactor2 = compute_prefactor_A0(first_entry)
        # Add -1 factor if second string has Y
        if "final_state_GY" in second_entry:
            prefactor2 *= -1
        
        return [prefactor1, prefactor2]

    prefactor_lists = []
    
    # Process each nsq group
    for nsq_group in combined_lists:
        nsq_prefactors = []
        
        # Process each pair in the group
        for pair in nsq_group:
            prefactors = get_prefactors(pair)
            nsq_prefactors.append(prefactors)
            
        prefactor_lists.append(nsq_prefactors)
    
    return prefactor_lists

def get_moms_and_prefacs_A2(x):
    A0=[]
    A1=[]
    prefA0=[]
    prefA1=[]
    for i in range(5):
        A0.append([sublist[0] for sublist in combine_momentum_lists()[i]])
        A1.append([sublist[1] for sublist in combine_momentum_lists()[i]])#
        prefA0.append([sublist[0] for sublist in compute_combined_prefactors(x)[i]])
        prefA1.append([sublist[1] for sublist in compute_combined_prefactors(x)[i]])
    return A0,prefA0,A1,prefA1
print(get_moms_and_prefacs_A2(10))