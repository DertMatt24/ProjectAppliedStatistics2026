library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;


ENTITY project_reti_logiche IS
    PORT (
        i_clk : in STD_LOGIC;
        i_rst : in STD_LOGIC;
        i_start : in STD_LOGIC;
        i_add : in std_logic_vector(15 downto 0);
        
        o_done : out STD_LOGIC;
        o_mem_addr : out STD_LOGIC_VECTOR (15 downto 0);
        i_mem_data : in STD_LOGIC_VECTOR (7 downto 0);
        o_mem_data : out STD_LOGIC_VECTOR (7 downto 0);
        o_mem_we : out STD_LOGIC;
        o_mem_en : out STD_LOGIC
    );
END project_reti_logiche;

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

ENTITY convolutor IS
    PORT ( 
        i_ftype : in STD_LOGIC;  
        
        i_c0: in SIGNED(7 DOWNTO 0);
        i_c1: in SIGNED(7 DOWNTO 0);
        i_c2: in SIGNED(7 DOWNTO 0);
        i_c3: in SIGNED(7 DOWNTO 0);
        i_c4: in SIGNED(7 DOWNTO 0);
        i_c5: in SIGNED(7 DOWNTO 0);
        i_c6: in SIGNED(7 DOWNTO 0);
                
        i_d0: in SIGNED(7 DOWNTO 0);
        i_d1: in SIGNED(7 DOWNTO 0);
        i_d2: in SIGNED(7 DOWNTO 0);
        i_d3: in SIGNED(7 DOWNTO 0);
        i_d4: in SIGNED(7 DOWNTO 0);
        i_d5: in SIGNED(7 DOWNTO 0);
        i_d6: in SIGNED(7 DOWNTO 0);
        
        i_clk: in STD_LOGIC;
        o_data : out SIGNED(17 DOWNTO 0)
    );
END convolutor;

ARCHITECTURE Behavioural of convolutor is

SIGNAL s_1: SIGNED(19 downto 0);
SIGNAL s_2: SIGNED(19 downto 0);
SIGNAL s_3: SIGNED(19 downto 0);
SIGNAL s3_or_zero: SIGNED(19 downto 0);

SIGNAL s_4: SIGNED (19  downto 0);
SIGNAL s_5: SIGNED (19  downto 0);

BEGIN
-- Compute both 
    
    s_1 <= resize((resize(i_c1, 9) * resize(i_d1, 9)) + (resize(i_c2, 9) * resize(i_d2, 9)), 20);
    s_2 <= resize((resize(i_c4, 9) * resize(i_d4, 9)) + (resize(i_c5, 9) * resize(i_d5, 9)), 20);
    s_3 <= resize((resize(i_c6, 9) * resize(i_d6, 9)) + (resize(i_c0, 9) * resize(i_d0, 9)), 20);
    s3_or_zero <= s_3 WHEN i_ftype = '1' ELSE (others => '0'); 
    s_4 <= resize(s_1 + s_2, 20);
    s_5 <= resize(s3_or_zero + (i_c3 * i_d3), 20);
    o_data <= resize(s_4 + s_5, 18);
    
END Behavioural;

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity shift_register is
    PORT (
        i_clk : in STD_LOGIC;
        i_en : in STD_LOGIC;
        i_zero: in STD_LOGIC;
        i_val : in STD_LOGIC_VECTOR(7 DOWNTO 0);
        i_rst : in STD_LOGIC;
        
        o_d0 : out SIGNED(7 DOWNTO 0);
        o_d1 : out SIGNED(7 DOWNTO 0);
        o_d2 : out SIGNED(7 DOWNTO 0);
        o_d3 : out SIGNED(7 DOWNTO 0);
        o_d4 : out SIGNED(7 DOWNTO 0);
        o_d5 : out SIGNED(7 DOWNTO 0);
        o_d6 : out SIGNED(7 DOWNTO 0)
    );
end shift_register;

architecture Behavioral of shift_register is

-- Implicitly declare these 7 registers
SIGNAL s_0: SIGNED(7 DOWNTO 0);
SIGNAL s_1: SIGNED(7 DOWNTO 0);
SIGNAL s_2: SIGNED(7 DOWNTO 0);
SIGNAL s_3: SIGNED(7 DOWNTO 0);
SIGNAL s_4: SIGNED(7 DOWNTO 0);
SIGNAL s_5: SIGNED(7 DOWNTO 0);
SIGNAL s_6: SIGNED(7 DOWNTO 0);

BEGIN

shift_down: PROCESS(i_clk, i_rst, i_en, i_zero)
VARIABLE value : SIGNED(7 DOWNTO 0);
BEGIN
IF (i_rst = '1') THEN
    s_0 <= (others => '0');
    s_1 <= (others => '0');
    s_2 <= (others => '0');
    s_3 <= (others => '0');
    s_4 <= (others => '0');
    s_5 <= (others => '0');
    s_6 <= (others => '0');
-- Make the component sensitive on the rising edge of the clock.
ELSIF (i_clk'event AND i_clk = '1' AND i_en = '1') THEN

    s_6 <= s_5;
    s_5 <= s_4;
    s_4 <= s_3;
    s_3 <= s_2;
    s_2 <= s_1;
    s_1 <= s_0;
    IF (i_zero = '0') THEN
        value := SIGNED(i_val);
    ELSE
        value := (others => '0');
    END IF;
    s_0 <= value;
ELSE
    NULL;
END IF;
END PROCESS shift_down;

-- Cheap hack ( o_d0 are outputs and cannot be read so we need this)
o_d0 <= s_0;
o_d1 <= s_1;
o_d2 <= s_2;
o_d3 <= s_3;
o_d4 <= s_4;
o_d5 <= s_5;
o_d6 <= s_6;

END Behavioral;

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

ENTITY normalizer IS
    PORT (
        i_ftype : in STD_LOGIC;
        i_data : in SIGNED (17 downto 0);
        i_clk : in STD_LOGIC;
        o_capped : out SIGNED (7 downto 0)
    );
END normalizer; 

ARCHITECTURE Behavioral OF normalizer IS

-- The values used to cap the output to an eight bit number
CONSTANT MAX_UPPER_SIGNED_BYTE : signed (7 downto 0) := TO_SIGNED(127, 8);
CONSTANT MIN_LOWER_SIGNED_BYTE : signed (7 downto 0) := TO_SIGNED(-128, 8);

-- The signals we use to take the sums
SIGNAL shifted_data_shared:  signed(15 downto 0);
SIGNAL shifted_data_ord3:  signed(15 downto 0);

SIGNAL data_out : signed(15 downto 0);
SIGNAL before_correction : signed(15 downto 0);
SIGNAL correction_amount : signed(3 DOWNTO 0);

BEGIN


-- Synchronize the activity to the clock

-- Compute first the common factors.
-- sords := data / 1024 + data / 64
shifted_data_shared <= resize(shift_right(SIGNED(i_data), 10) + shift_right(SIGNED(i_data), 6), 16);    
-- We have to compute two other sums, so the latency is tripled!
-- sord3 := data / 16 + data / 256
shifted_data_ord3 <= resize(shift_right(SIGNED(i_data), 4) + shift_right(SIGNED(i_data), 8), 16);
-- data_out := shifted_data_ord3 + shifted_data_shared;
-- data_out := shifted_data_shared;

correction_amount <= 
    to_signed(+2, 4) WHEN i_data < 0 AND i_ftype = '1' ELSE
    to_signed(+4, 4) WHEN I_data < 0 AND i_ftype = '0' ELSE
    to_signed(+0, 4);
-- Adjustments int he event that our data to be truncated was 
-- negative.
before_correction <= 
    shifted_data_shared + shifted_data_ord3 WHEN i_ftype = '0' ELSE
    shifted_data_shared WHEN i_ftype = '1';
    
data_out <= before_correction + correction_amount;
-- Now cap the output data to an 8 bit integer.
-- Cap the value to positive 127

o_capped <= 
    MAX_UPPER_SIGNED_BYTE WHEN (data_out > 127) ELSE
    MIN_LOWER_SIGNED_BYTE WHEN (data_out < -128) ELSE
    data_out(7 downto 0);
    
END Behavioral;

ARCHITECTURE Behavioral OF project_reti_logiche IS

-- Handle normalization and the capping of data to a signed byte.
    COMPONENT normalizer IS
    PORT (
        i_ftype : in STD_LOGIC;
        i_data : in SIGNED (17 downto 0);
        i_clk : in STD_LOGIC;        
        o_capped : out SIGNED (7 downto 0)
    );
    END COMPONENT;

-- Handle the computation of the kernel itself.
    COMPONENT convolutor IS
        PORT ( 
            i_ftype : in STD_LOGIC;  
            
            i_c0: in SIGNED(7 DOWNTO 0);
            i_c1: in SIGNED(7 DOWNTO 0);
            i_c2: in SIGNED(7 DOWNTO 0);
            i_c3: in SIGNED(7 DOWNTO 0);
            i_c4: in SIGNED(7 DOWNTO 0);
            i_c5: in SIGNED(7 DOWNTO 0);
            i_c6: in SIGNED(7 DOWNTO 0);
                    
            i_d0: in SIGNED(7 DOWNTO 0);
            i_d1: in SIGNED(7 DOWNTO 0);
            i_d2: in SIGNED(7 DOWNTO 0);
            i_d3: in SIGNED(7 DOWNTO 0);
            i_d4: in SIGNED(7 DOWNTO 0);
            i_d5: in SIGNED(7 DOWNTO 0);
            i_d6: in SIGNED(7 DOWNTO 0);
            
            i_clk: in STD_LOGIC;
            o_data : out SIGNED(17 downto 0)
        );
    END COMPONENT;

    COMPONENT shift_register IS
    PORT (
        i_zero : in STD_LOGIC;
        i_clk : in STD_LOGIC;
        i_en : in STD_LOGIC;
        i_val : in STD_LOGIC_VECTOR(7 DOWNTO 0);
        i_rst: in STD_LOGIC;
        
        o_d0 : out SIGNED(7 DOWNTO 0);
        o_d1 : out SIGNED(7 DOWNTO 0);
        o_d2 : out SIGNED(7 DOWNTO 0);
        o_d3 : out SIGNED(7 DOWNTO 0);
        o_d4 : out SIGNED(7 DOWNTO 0);
        o_d5 : out SIGNED(7 DOWNTO 0);
        o_d6 : out SIGNED(7 DOWNTO 0)
    );
    END COMPONENT;
    
TYPE signed_array IS ARRAY(6 DOWNTO 0) OF SIGNED(7 DOWNTO 0);

-- DESCRIPTOR ------------------------------|
-- The amount of data to be processed       |
SIGNAL data_amt: UNSIGNED(15 downto 0);     -- K1, K2
-- |                                        |
-- The type of the filter                   |    
SIGNAL f_type: STD_LOGIC;                   -- S
-- |                                        |
-- The coefficients                         |
SIGNAL coefficients : signed_array;         -- C1, C2, ... C7 OR C8, ..., C14

TYPE header IS ARRAY(16 DOWNTO 0) OF STD_LOGIC_VECTOR(7 DOWNTO 0);

-- HEADER & UTILITY ------------------------------------
-- Current portion of the data                          |
SIGNAL data_pool : signed_array;                        
-- |                                                    |
-- Shift enable for the shifting register.              |
SIGNAL enable_shift: STD_LOGIC;         -- MAPPED
-- |                                                    |    
-- The result of applying the convolution               |
SIGNAL unnormalized_kernel: SIGNED(17 downto 0);        -- MAPPED
-- |                                                    |
-- The data, ready to be written to memory              |
SIGNAL processed_data: SIGNED(7 downto 0);              

-- Reset the shift register synchronously. 
SIGNAL reset_reg: STD_LOGIC := '0';     -- MAPPED

-- Read zero into the shift register synchronously.
SIGNAL read_zero: STD_LOGIC;


TYPE inner_memory_state IS (START, K0, K1, S, CS, STOP, WAIT_MEM_OFFSET);
TYPE rotating_reg_state IS (START, ONE, TWO, THREE, FOUR, STOP, WAIT_MEM_OFFSET);
TYPE operate_state IS (START, READ_ONLY, NOW_READ, WAIT_R, NOW_WRITE, WAIT_W, STOP);
TYPE machine_state IS (HEADER_READ, INITIALIZATION, OPERATE, DONE);

TYPE bounded_int IS RANGE 17 DOWNTO 0;

BEGIN

KERNEL: convolutor 
PORT MAP( 
    i_c0 => coefficients(6),
    i_c1 => coefficients(5),
    i_c2 => coefficients(4),
    i_c3 => coefficients(3),
    i_c4 => coefficients(2),
    i_c5 => coefficients(1),
    i_c6 => coefficients(0),
    i_ftype => f_type,
    
    i_d0 => data_pool(0),
    i_d1 => data_pool(1),
    i_d2 => data_pool(2),
    i_d3 => data_pool(3),
    i_d4 => data_pool(4),
    i_d5 => data_pool(5),
    i_d6 => data_pool(6),
    
    i_clk => i_clk,
    o_data => unnormalized_kernel
);

OUT_NORMALIZER: normalizer
PORT MAP(
    i_data => unnormalized_kernel,
    i_clk => i_clk,
    i_ftype => f_type,    
    o_capped => processed_data
 );

SH_REGISTER: shift_register
PORT MAP(
    i_clk => i_clk,
    i_en => enable_shift,
    i_zero => read_zero,
    i_val => i_mem_data,
    i_rst => reset_reg,
    
    o_d0 => data_pool(0),
    o_d1 => data_pool(1),
    o_d2 => data_pool(2),
    o_d3 => data_pool(3),
    o_d4 => data_pool(4),
    o_d5 => data_pool(5),
    o_d6 => data_pool(6)
);

compute_func: PROCESS(i_clk, i_rst)

VARIABLE read_address: UNSIGNED(15 downto 0) ;
VARIABLE write_address: UNSIGNED(15 downto 0);

VARIABLE max_read_address: UNSIGNED(15 DOWNTO 0);
VARIABLE max_write_address: UNSIGNED(15 DOWNTO 0);

-- Index used when reading the header.
VARIABLE head_index: bounded_int; 

-- States for the operation of the FSA, including the wait states. 
VARIABLE header_state: inner_memory_state;
VARIABLE state_while_waiting: inner_memory_state;

VARIABLE reg_state: rotating_reg_state;
VARIABLE reg_state_while_waiting: rotating_reg_state;

VARIABLE op_state: operate_state;
VARIABLE total_state: machine_state;

-- The value where the exit of the multiplier is to be saved. 
VARIABLE prev_data: STD_LOGIC_VECTOR(7 DOWNTO 0);

-- Marks when the component is ready to initiating writes
VARIABLE write_ready: STD_LOGIC;

BEGIN

IF ( i_rst = '1' OR i_start = '0' ) THEN
    
    -- Set an initial value for this index.
    head_index   := 0;
    
    -- Initialize the entirety of the states. 
    reg_state    := START;
    header_state := START;
    op_state     := START;
    -- Initialize the entire machine
    total_state  := HEADER_READ;
    -- Assert done as zero.
    o_done       <= '0';
    o_mem_we     <= '0';
    o_mem_en     <= '0';
    enable_shift <= '0';
    
-- Synchronize the remaining component to the clock cycles.
ELSIF (rising_edge(i_clk) AND i_start = '1') THEN
    
    IF (total_state = HEADER_READ) THEN
    CASE header_state IS
        WHEN WAIT_MEM_OFFSET =>
            -- A common wait state.
            header_state := state_while_waiting;
        WHEN START =>
            -- Begin reading the data into the header variables one by one.
            -- Every time, allow the memory one cycle of wait time by jumping
            -- back and forth from the wait state.
            
            o_mem_en    <= '1';
            o_mem_addr  <= STD_LOGIC_VECTOR(i_add);
            
            read_address        := unsigned(i_add) + 1;

            header_state        := WAIT_MEM_OFFSET;
            state_while_waiting := K0;
            reset_reg    <= '1';

        WHEN K0 =>
        
            data_amt(15 DOWNTO 8) <= unsigned(i_mem_data);
            o_mem_addr            <= STD_LOGIC_VECTOR(read_address);
            
            read_address        := read_address + 1;
           
            header_state        := WAIT_MEM_OFFSET;
            state_while_waiting := K1;
        
        WHEN K1 =>
        
            data_amt(7 DOWNTO 0) <= unsigned(i_mem_data);
            
            o_mem_addr <= STD_LOGIC_VECTOR(read_address);

            header_state        := WAIT_MEM_OFFSET;
            state_while_waiting := S;
            read_address        := read_address + 1;
            reset_reg    <= '0';

        WHEN S =>
        
            f_type <= STD_LOGIC(i_mem_data(0));
            -- Depending on the type of filter, we might have to jump the 
            -- 7 bytes to get to the correct coefficients.
            if (STD_LOGIC(i_mem_data(0)) = '1') THEN
                read_address := read_address + 7;
            END IF;
            
            o_mem_addr <= STD_LOGIC_VECTOR(read_address);
            read_address        := read_address + 1;
            header_state        := WAIT_MEM_OFFSET;
            state_while_waiting := CS;
            
        WHEN CS => 
            -- e.g. 1, -9, 45, 0, -45, 9, -1
            -- Load the coefficients one by one. 
            coefficients(INTEGER(head_index)) <= signed(i_mem_data);
            o_mem_addr                        <= STD_LOGIC_VECTOR(read_address);
            read_address    := read_address + 1;
            head_index      := head_index + 1;

            IF (head_index = 7) THEN
                -- Notice that at the end, the value of read_address
                -- is equal to the value of the first memory cell.
                header_state := STOP;
            ELSE
                header_state        := WAIT_MEM_OFFSET;
                state_while_waiting := CS;
            END IF;
           
        WHEN STOP =>
        
            -- Reset the register so that at the next clock cycle
            -- we'll be ready to write.
            reset_reg <= '1';
            o_mem_en  <= '0';
            total_state := INITIALIZATION;
            
        END CASE;
        
    ELSIF (total_state = INITIALIZATION) THEN
    -- Note: Whenever we are done with the last element in the header, the first
    -- element of the data will already be read!
        CASE reg_state IS
            WHEN WAIT_MEM_OFFSET =>
                -- Note that we enable the shift in this wait state.
                enable_shift <= '1';
                reg_state := reg_state_while_waiting;
                
            WHEN START =>
                -- Initialize the reading address and begin reading.
                read_address := UNSIGNED(i_add) + 17;
                reset_reg <= '0';
                o_mem_en  <= '1';
                read_zero <= '0';
                reg_state := ONE;
                
            WHEN ONE | TWO => 
            
                -- Stop shifting, the next clock cycle we're going to be in the middle of a
                -- write sequence. 
                enable_shift <= '0';
                o_mem_addr   <= STD_LOGIC_VECTOR(read_address);
                read_address := read_address + 1;
                 
                IF (reg_state = ONE) THEN
                    reg_state_while_waiting := TWO;
                ELSE 
                    reg_state_while_waiting := THREE;
                END IF;
                reg_state := WAIT_MEM_OFFSET;
                
            WHEN THREE =>
            
                enable_shift <= '0';
                o_mem_addr   <= STD_LOGIC_VECTOR(read_address);
                read_address := read_address + 1;
                reg_state    := STOP;
                
            WHEN STOP =>
                -- Begin operating. Leave the shift enabled to write the value
                -- read just previously in THREE. 
                total_state := OPERATE;
                enable_shift <= '1';
                
            WHEN OTHERS => 
                NULL;
        END CASE;
    
    ELSIF (total_state = OPERATE) THEN
    CASE op_state IS
    
        WHEN START =>
            -- Initialize all utility variables for the reading 
            -- and processing cycle.
            enable_shift <= '0';
            op_state          := READ_ONLY;
            write_ready       := '0';
            write_address     := UNSIGNED(i_add) + data_amt + 17;
            max_read_address  := write_address;
            max_write_address := write_address + data_amt;
            
        WHEN READ_ONLY => 
            
            o_mem_addr <= STD_LOGIC_VECTOR(read_address);
            read_address := read_address + 1;
            op_state     := WAIT_R;
            
        WHEN NOW_READ =>
            
            o_mem_we   <= '0';
            IF ( read_address < max_read_address ) THEN
                o_mem_addr <= STD_LOGIC_VECTOR(read_address);
                read_address := read_address + 1;
            ELSE
                -- If the memory to read has exhausted, start padding with virtual
                -- zero
                read_zero <= '1';
            END IF;
            
            prev_data    := STD_LOGIC_VECTOR(processed_data);
            op_state     := WAIT_R;
            write_ready  := '1';
            
        WHEN WAIT_R =>
        
            enable_shift <= '1';
            op_state := NOW_WRITE;
            
        WHEN NOW_WRITE =>
            
            enable_shift <= '0';
            op_state := WAIT_W;
            IF ( write_ready = '1' ) THEN
                IF ( write_address < max_write_address ) THEN
                    o_mem_we    <= '1';
                    o_mem_addr  <= STD_LOGIC_VECTOR(write_address);
                    o_mem_data  <= prev_data;
                    write_address := write_address + 1;
                ELSE
                    -- If we have reached the end write address, stop the machine. 
                    op_state := STOP;
                END IF;
            END IF;
        
        WHEN WAIT_W =>
        
            op_state := NOW_READ;
            
        WHEN STOP =>
            -- Stop the machine and assert the o_done signal. 
            total_state := DONE;
            o_done <= '1';
            o_mem_en <= '0';
    END CASE;
    
    END IF;
    
END IF;

END PROCESS;

END Behavioral;